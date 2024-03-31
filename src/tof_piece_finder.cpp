#include <ament_index_cpp/get_package_share_directory.hpp>
#include <filesystem>
#include <limits>
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

namespace fs = std::filesystem;
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
    it_ = make_unique<image_transport::ImageTransport>(node);

    // Load the gripper mask.
    fs::path share_dir(ament_index_cpp::get_package_share_directory("tof_piece_finder"));
    fs::path gripper_mask_file(params_->filter.gripper_mask_file);
    fs::path full_path = share_dir / gripper_mask_file;
    cv::Mat gripper_mask_u8 = cv::imread(full_path, cv::IMREAD_GRAYSCALE);
    gripper_mask_ = cv::Mat(gripper_mask_u8.rows, gripper_mask_u8.cols, CV_32FC1);
    for (int row = 0; row < gripper_mask_.rows; ++row) {
      for (int col = 0; col < gripper_mask_.cols; ++col) {
        if (gripper_mask_u8.at<uint8_t>(row, col) == 0)
          gripper_mask_.at<float>(row, col) = numeric_limits<float>::infinity();
        else
          gripper_mask_.at<float>(row, col) = 1.f;
      }
    }

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
    binary_image_pub_ =
        make_unique<image_transport::Publisher>(it_->advertise(params_->binary_image_topic, 1));
    annotated_image_pub_ =
        make_unique<image_transport::Publisher>(it_->advertise(params_->annotated_image_topic, 1));

    // Setup the camera subscribers.
    const auto depth_callback = bind(&TofPieceFinder::depth_image_callback, this, _1, _2);
    depth_camera_sub_ = make_unique<image_transport::CameraSubscriber>(
        it_->subscribeCamera(params_->depth_camera_base_topic, 1, depth_callback));
    const auto ir_callback = bind(&TofPieceFinder::ir_image_callback, this, _1, _2);
    ir_camera_sub_ = make_unique<image_transport::CameraSubscriber>(
        it_->subscribeCamera(params_->ir_camera_base_topic, 1, ir_callback));

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
   * Callback for the depth image.
   *
   * @param[in] image The depth image message.
   * @param[in] cinfo The camera info message.
   */
  void depth_image_callback(const sensor_msgs::msg::Image::ConstSharedPtr& image,
                            const sensor_msgs::msg::CameraInfo::ConstSharedPtr& cinfo)
  {
    // Convert the image message to a cv::Mat.
    cv_bridge::CvImagePtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::TYPE_32FC1);
    } catch (cv_bridge::Exception& e) {
      RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
      return;
    }
    cv_ptr->image.copyTo(depth_img_);
    depth_img_stamp_ = rclcpp::Time(image->header.stamp.sec, image->header.stamp.nanosec);

    // Copy camera info.
    camera_frame_ = cinfo->header.frame_id;
    fx_ = cinfo->k[0];
    fy_ = cinfo->k[4];
    cx_ = cinfo->k[2];
    cy_ = cinfo->k[5];

    if (depth_img_stamp_ == ir_img_stamp_) process_images();
  }

  /**
   * Callback for the ir image.
   *
   * @param[in] image The ir image message.
   * @param[in] cinfo The camera info message.
   */
  void ir_image_callback(const sensor_msgs::msg::Image::ConstSharedPtr& image,
                         const sensor_msgs::msg::CameraInfo::ConstSharedPtr& cinfo)
  {
    // Convert the image message to a cv::Mat.
    cv_bridge::CvImagePtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::TYPE_32FC1);
    } catch (cv_bridge::Exception& e) {
      RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
      return;
    }
    cv_ptr->image.copyTo(ir_img_);
    ir_img_stamp_ = rclcpp::Time(image->header.stamp.sec, image->header.stamp.nanosec);

    // Copy camera info.
    camera_frame_ = cinfo->header.frame_id;
    fx_ = cinfo->k[0];
    fy_ = cinfo->k[4];
    cx_ = cinfo->k[2];
    cy_ = cinfo->k[5];

    if (depth_img_stamp_ == ir_img_stamp_) process_images();
  }

  /**
   * Process the images and publishes a sparse point cloud message, where each point represents a
   * piece on the board. It will also publish a thresholded image for debugging.
   */
  void process_images()
  {
    auto now = rclcpp::Clock().now();

    // Check for parameter updates.
    if (param_listener_->is_old(*params_)) {
      params_ = make_unique<Params>(param_listener_->get_params());
    }

    // Apply a gaussian blur to the images.
    static cv::Mat depth_blurred;
    int blur_size = params_->filter.blur_size;
    cv::GaussianBlur(depth_img_, depth_blurred, cv::Size(blur_size, blur_size), 0);

    // Apply the gripper mask to the image.
    cv::Mat depth_masked;
    if (gripper_mask_.data == nullptr)
      depth_masked = depth_blurred;
    else
      depth_masked = depth_blurred.mul(gripper_mask_);

    // Remove all pixels that have too low of an IR intensity.
    // float min_intensity = numeric_limits<float>::infinity();
    // float max_intensity = 0;
    for (int i = 0; i < depth_masked.rows; i++) {
      for (int j = 0; j < depth_masked.cols; j++) {
        const float intensity = ir_img_.at<float>(i, j);
        // if (intensity < min_intensity) min_intensity = intensity;
        // if (intensity > max_intensity) max_intensity = intensity;
        if (intensity < 8000) depth_masked.at<float>(i, j) = numeric_limits<float>::infinity();
      }
    }
    // RCLCPP_INFO(node->get_logger(), "Intensity: %0.3f -> %0.3f", min_intensity, max_intensity);

    // Create a sorted vector of distances from the image, filtering out the gripper mask and any
    // distances outside the min and max limits.
    vector<float> distances;
    for (int i = 0; i < depth_masked.rows; i++) {
      for (int j = 0; j < depth_masked.cols; j++) {
        float distance = depth_masked.at<float>(i, j);
        if (distance < params_->filter.limits.max_distance &&
            distance > params_->filter.limits.min_distance) {
          distances.push_back(distance);
        }
      }
    }
    sort(distances.begin(), distances.end());
    if (distances.size() < 2) {
      // RCLCPP_WARN(get_logger(), "Cannot find anything");
      return;
    }

    // Find the threshold distance. This is either the median distance in the filtered image, or
    // a constant offset from the nearest distance. The nearer of the two is used.
    float top_5pct = distances.size() * 0.05;
    float nearest_distance_cutoff = distances[top_5pct] + params_->filter.max_height_difference;
    float median_distance_cutoff = distances[distances.size() / 2] - 0.01f;
    float threshold_distance = min(nearest_distance_cutoff, median_distance_cutoff);

    // Create a binary image from the threshold distance. We remove any pixels that are further away
    // than the threshold distance. This should remove the chessboard and any other objects that are
    // further away than the pieces.
    cv::Mat binary;
    cv::threshold(depth_masked, binary, threshold_distance, 255, cv::THRESH_BINARY_INV);
    binary.convertTo(binary, CV_8UC1);

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

#if 0
    // Find the highest point of every component.
    vector<array<int, 2>> highest_points;
    for (size_t i = 0; i < valid_labels.size(); ++i) {
      int label = valid_labels[i];
      float best = numeric_limits<float>::infinity();
      int best_row = 0;
      int best_col = 0;
      for (int row = 0; row < depth_masked.rows; ++row) {
        for (int col = 0; col < depth_masked.cols; ++col) {
          float value = depth_masked.at<float>(row, col);
          if (labels.at<int32_t>(row, col) == label && value < best) {
            best_row = row;
            best_col = col;
            best = value;
          }
        }
      }
      highest_points.emplace_back(array<int, 2>({ best_col, best_row }));
    }
#endif

    // Create a copy of the depth image to draw points on.
    cv::Mat depth_annotated_grey, depth_annotated;
    cv::normalize(ir_img_, depth_annotated_grey, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::cvtColor(depth_annotated_grey, depth_annotated, cv::COLOR_GRAY2RGB);

    // Create a point cloud message from the connected components.
    int n_points = valid_labels.size();
    pointcloud_msg_.header.stamp = now;
    pointcloud_msg_.header.frame_id = camera_frame_;
    pointcloud_msg_.data.clear();
    pointcloud_msg_.data.reserve(n_points * 12);
    pointcloud_msg_.width = n_points;
    pointcloud_msg_.row_step = n_points * 12;
    for (int i = 0; i < n_points; i++) {
      const double col_d = centroids.at<double>(valid_labels[i], 0);
      const double row_d = centroids.at<double>(valid_labels[i], 1);
      int col = round(col_d);
      int row = round(row_d);
      if (col < 0) col = 0;
      if (row < 0) row = 0;
      if (col > depth_masked.cols - 1) col = depth_masked.cols - 1;
      if (row > depth_masked.rows - 1) col = depth_masked.rows - 1;

      // Draw point on annotated image.
      cv::circle(depth_annotated, cv::Point2i(col, row), 2, cv::Scalar(255, 0, 0), -1);

      // Convert the pixel to a 3D point.
      float point[3];
      point[0] = -(((cx_ - col)) / fx_) * depth_masked.at<float>(row, col);  // X
      point[1] = -(((cy_ - row)) / fy_) * depth_masked.at<float>(row, col);  // Y
      point[2] = depth_masked.at<float>(row, col);                           // Z

      // Add the point to the point cloud message.
      uint8_t* data = reinterpret_cast<uint8_t*>(point);
      pointcloud_msg_.data.insert(pointcloud_msg_.data.end(), data, data + 12);
    }

    // Publish the point cloud message.
    pointcloud_pub_->publish(pointcloud_msg_);

    // Publish the thresholded image.
    cv_bridge::CvImage bin_msg;
    bin_msg.header.stamp = now;
    bin_msg.header.frame_id = camera_frame_;
    bin_msg.encoding = sensor_msgs::image_encodings::MONO8;
    bin_msg.image = binary;
    binary_image_pub_->publish(bin_msg.toImageMsg());

    // Publish the annotated image.
    cv_bridge::CvImage ann_msg;
    ann_msg.header.stamp = now;
    ann_msg.header.frame_id = camera_frame_;
    ann_msg.encoding = sensor_msgs::image_encodings::RGB8;
    ann_msg.image = depth_annotated;
    annotated_image_pub_->publish(ann_msg.toImageMsg());
  }

  unique_ptr<ParamListener> param_listener_;        //< Parameter listener for this node.
  unique_ptr<Params> params_;                       //< The parameters for this node.
  unique_ptr<image_transport::ImageTransport> it_;  //< Image transport for this node.
  unique_ptr<image_transport::Publisher> binary_image_pub_;
  unique_ptr<image_transport::Publisher> annotated_image_pub_;
  unique_ptr<image_transport::CameraSubscriber> depth_camera_sub_;
  unique_ptr<image_transport::CameraSubscriber> ir_camera_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_pub_;

  cv::Mat depth_img_;
  cv::Mat ir_img_;
  rclcpp::Time depth_img_stamp_;
  rclcpp::Time ir_img_stamp_;
  std::string camera_frame_;
  double fx_;  // Focal length in x
  double fy_;  // Focal length in y
  double cx_;  // Optical center in x
  double cy_;  // Optical center in y

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
