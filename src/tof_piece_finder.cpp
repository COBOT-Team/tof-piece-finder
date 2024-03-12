#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>

#include "cv_bridge/cv_bridge.h"
#include "image_transport/image_transport.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "rclcpp/logging.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "tof_piece_finder_params.hpp"

using namespace std;
using namespace tof_piece_finder;
using std::placeholders::_1;
using std::placeholders::_2;

static const bool IS_BIG_ENDIAN = false;  // TODO: Determine this programatically.

class TofPieceFinder
{
public:
  rclcpp::Node::SharedPtr node;  //< ROS2 node.

  /**
   * Get the logger for the node.
   *
   * @return The logger for the node.
   */
  rclcpp::Logger get_logger() const { return node->get_logger(); }

  /**
   * Construct a new Tof Piece Finder object.
   */
  explicit TofPieceFinder()
  {
    node = rclcpp::Node::make_shared("tof_piece_finder");
    param_listener_ = make_unique<ParamListener>(node);
    params_ = make_unique<Params>(param_listener_->get_params());
    gripper_mask_ = cv::imread(params_->filter.gripper_mask_file, cv::IMREAD_GRAYSCALE);
    it_ = make_unique<image_transport::ImageTransport>(node);

    // Setup the point cloud message.
    pointcloud_msg_.height = 1;
    pointcloud_msg_.fields.reserve(4);
    pointcloud_msg_.fields.emplace_back(
        create_point_field("x", 0, sensor_msgs::msg::PointField::FLOAT32, 1));
    pointcloud_msg_.fields.emplace_back(
        create_point_field("y", 4, sensor_msgs::msg::PointField::FLOAT32, 1));
    pointcloud_msg_.fields.emplace_back(
        create_point_field("z", 8, sensor_msgs::msg::PointField::FLOAT32, 1));
    pointcloud_msg_.is_bigendian = IS_BIG_ENDIAN;
    pointcloud_msg_.point_step = 12;
    pointcloud_msg_.is_dense = false;

    // Setup the publishers.
    pointcloud_pub_ =
        node->create_publisher<sensor_msgs::msg::PointCloud2>(params_->points_topic, 10);
    image_pub_ =
        make_unique<image_transport::Publisher>(it_->advertise(params_->pieces_image_topic, 1));

    // Setup the camera subscriber.
    auto bound_callback = bind(&TofPieceFinder::image_callback, this, _1, _2);
    camera_sub_ = make_unique<image_transport::CameraSubscriber>(
        it_->subscribeCamera(params_->camera_base_topic, 1, bound_callback));

    RCLCPP_INFO(get_logger(), "Tof Piece Finder node started");
  }

private:
  /**
   * Create a point field message. This is used to create the fields for the point cloud message.
   *
   * @param[in] name The name of the field.
   * @param[in] offset The offset of the field in the point cloud message.
   * @param[in] datatype The datatype of the field.
   * @param[in] count The number of elements in the field.
   * @return The point field message.
   */
  sensor_msgs::msg::PointField create_point_field(const std::string& name, const uint32_t offset,
                                                  const uint8_t datatype, const uint32_t count)
  {
    sensor_msgs::msg::PointField field;
    field.name = name;
    field.offset = offset;
    field.datatype = datatype;
    field.count = count;
    return field;
  }

  /**
   * Callback for the depth image. This function processes the depth image and publishes a sparse
   * point cloud message, where each point represents a piece on the board. It will also publish
   * a thresholded image for debugging.
   *
   * @param[in] image The depth image message.
   * @param[in] cinfo The camera info message.
   */
  void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr& image,
                      const sensor_msgs::msg::CameraInfo::ConstSharedPtr& cinfo)
  {
    auto now = rclcpp::Clock().now();

    // Check for parameter updates.
    if (param_listener_->is_old(*params_)) {
      params_ = make_unique<Params>(param_listener_->get_params());
      gripper_mask_ = cv::imread(params_->filter.gripper_mask_file, cv::IMREAD_GRAYSCALE);
    }

    // Convert the image message to a cv::Mat.
    cv_bridge::CvImagePtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::TYPE_32FC1);
    } catch (cv_bridge::Exception& e) {
      RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
      return;
    }

    // Extract important camera parameters.
    string camera_frame = cinfo->header.frame_id;
    double fx = cinfo->k[0];  // Focal length in x
    double fy = cinfo->k[4];  // Focal length in y
    double cx = cinfo->k[2];  // Optical center in x
    double cy = cinfo->k[5];  // Optical center in y

    // Apply a gaussian blur to the image.
    static cv::Mat blurred;
    int blur_size = params_->filter.blur_size;
    cv::GaussianBlur(cv_ptr->image, blurred, cv::Size(blur_size, blur_size), 0);

    // Apply the gripper mask to the image.
    cv::Mat masked;
    cv::bitwise_and(blurred, gripper_mask_, masked);

    // Create a sorted vector of distances from the image, filtering out the gripper mask and any
    // distances outside the min and max limits.
    vector<float> distances;
    for (int i = 0; i < masked.rows; i++) {
      for (int j = 0; j < masked.cols; j++) {
        float distance = masked.at<float>(i, j);
        if (distance < params_->filter.limits.max_distance &&
            distance > params_->filter.limits.min_distance) {
          distances.push_back(distance);
        }
      }
    }
    sort(distances.begin(), distances.end());

    // TODO: Consider removing outliers from the distances.

    // Find the threshold distance. This is either the median distance in the filtered image, or
    // a constant offset from the nearest distance. The nearer of the two is used.
    float nearest_distance_cutoff = distances.front() + params_->filter.max_height_difference;
    float median_distance = distances[distances.size() / 2];
    float threshold_distance = min(nearest_distance_cutoff, median_distance);

    // Create a binary image from the threshold distance. We remove any pixels that are further away
    // than the threshold distance. This should remove the chessboard and any other objects that are
    // further away than the pieces.
    cv::Mat binary;
    cv::threshold(masked, binary, threshold_distance, 1, cv::THRESH_BINARY_INV);

    // Find connected components in the binary image. The first component is the background, so we
    // ignore it.
    cv::Mat labels, stats, centroids;
    int n_labels = cv::connectedComponentsWithStats(binary, labels, stats, centroids);

    // Filter out the connected components that are too small or too large.
    vector<int> valid_labels;
    for (int i = 1; i < n_labels; i++) {
      int area = stats.at<int>(i, cv::CC_STAT_AREA);
      if (area < params_->filter.limits.min_piece_size ||
          area > params_->filter.limits.max_piece_size) {
        continue;
      }
      valid_labels.push_back(i);
    }

    // Create a point cloud message from the connected components.
    int n_points = valid_labels.size();
    pointcloud_msg_.header.stamp = now;
    pointcloud_msg_.header.frame_id = camera_frame;
    pointcloud_msg_.data.clear();
    pointcloud_msg_.data.reserve(n_points * 12);
    pointcloud_msg_.width = n_points;
    pointcloud_msg_.row_step = n_points * 12;
    for (int i = 0; i < n_points; i++) {
      int label = valid_labels[i];
      int col = centroids.at<double>(label, 0);
      int row = centroids.at<double>(label, 1);

      // Convert the pixel to a 3D point.
      float point[3];
      point[0] = (((cx - col)) / fx) * masked.at<float>(row, col);  // X
      point[1] = (((cy - row)) / fy) * masked.at<float>(row, col);  // Y
      point[2] = masked.at<float>(row, col);                        // Z

      // Add the point to the point cloud message.
      uint8_t* data = reinterpret_cast<uint8_t*>(point);
      pointcloud_msg_.data.insert(pointcloud_msg_.data.end(), data, data + 12);
    }

    // Publish the point cloud message.
    pointcloud_pub_->publish(pointcloud_msg_);

    // Publish the thresholded image for debugging.
    cv_bridge::CvImage out_msg;
    out_msg.header.stamp = now;
    out_msg.header.frame_id = camera_frame;
    out_msg.encoding = sensor_msgs::image_encodings::MONO8;
    out_msg.image = binary;
    image_pub_->publish(out_msg.toImageMsg());
  }

  unique_ptr<ParamListener> param_listener_;        //< Parameter listener for this node.
  unique_ptr<Params> params_;                       //< The parameters for this node.
  unique_ptr<image_transport::ImageTransport> it_;  //< Image transport for this node.
  unique_ptr<image_transport::Publisher> image_pub_;
  unique_ptr<image_transport::CameraSubscriber> camera_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_pub_;

  cv::Mat gripper_mask_;
  sensor_msgs::msg::PointCloud2 pointcloud_msg_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = make_shared<TofPieceFinder>();
  rclcpp::spin(node->node);
  rclcpp::shutdown();
  return 0;
}
