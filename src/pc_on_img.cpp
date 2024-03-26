#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <image_transport/image_transport.h>
#include <pcl/point_types.h>
#include <pcl/range_image/range_image_spherical.h>
#include <pcl/filters/filter.h>
//#include <pcl/common/impl/transforms.hpp> // SWAN: transform
#include <opencv2/core/core.hpp>
#include <math.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Dense>

using namespace Eigen;
using namespace sensor_msgs;
using namespace message_filters;
using namespace std;

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
typedef pcl::PointCloud<pcl::PointXYZRGB> ColorPointCloud;

ros::Publisher pcOnimg_pub;
ros::Publisher pub;

// float maxlen = 10;
// float minlen = 0.1;
// float max_FOV = 1.6;
// float min_FOV = 0.9;
std::string imgTopic = "/velodyne_points";
std::string pcTopic = "/blackflys/image_raw";

// Extrinsic parameters
Eigen::MatrixXf C_T_L(4, 4);

// Intrinsic parameters
Eigen::MatrixXf M(3, 4);    // camera calibration matrix
Eigen::MatrixXf D(1, 5);    // camera distortion

// Image
cv::Mat orig_image;

// // ref: https://www.particleincell.com/2014/colormap/
// void setRainbowColor(const float value, int &R, int &G, int &B)
// {
//   // value in [0, 1]
//   float a = (1.f - value) / 0.25f;	//invert and group
//   int X = floorf(a);	//this is the integer part
//   int Y = floorf(255*(a-X)); //fractional part from 0 to 255
//   switch(X)
//   {
//       case 0: R = 255;   G = Y;     B = 0;   break;
//       case 1: R = 255-Y; G = 255;   B = 0;   break;
//       case 2: R = 0;     G = 255;   B = Y;   break;
//       case 3: R = 0;     G = 255-Y; B = 255; break;
//       case 4: R = 0;     G = 0;     B = 255; break;
//   }
// }

void getColor(float v, const float vmin, const float vmax,
              uint8_t &R, uint8_t &G, uint8_t &B)
{
  // default: white
  float r = 1.f, g = 1.f, b = 1.f;

  // clamp
  if (v < vmin) v = vmin;
  if (v > vmax) v = vmax;
  const float dv = vmax - vmin;

   if (v < (vmin + 0.25 * dv))
   {
      r = 0;
      g = 4 * (v - vmin) / dv;
   }
   else if (v < (vmin + 0.5 * dv))
   {
      r = 0;
      b = 1 + 4 * (vmin + 0.25 * dv - v) / dv;
   }
   else if (v < (vmin + 0.75 * dv))
   {
      r = 4 * (v - vmin - 0.5 * dv) / dv;
      b = 0;
   }
   else
   {
      g = 1 + 4 * (vmin + 0.75 * dv - v) / dv;
      b = 0;
   }

   R = 255 * r;
   G = 255 * g;
   B = 255 * b;
}

void reviseTransformation(Eigen::MatrixXf &L_R_C_ROS, Eigen::MatrixXf &L_t_C_ROS)
{
  // C_T_L: transform points from lidar coords to camera coords
  Eigen::MatrixXf C_ROS_R_L(3,3);
  Eigen::MatrixXf C_ROS_t_L(3,1);
  C_ROS_R_L = L_R_C_ROS.transpose();
  C_ROS_t_L = - C_ROS_R_L * L_t_C_ROS;  

  // translation from ROS camera to conventional camera
  //
  //      <ROS camera>               <camera>
  //
  //       (forward)    X                  z (forward)
  //           (up) Z   ^                  ^
  //                ^  /                  /
  //                | /                  /
  //                |/                  /
  // (left) Y <------                  --------> x (right)
  //                                   |
  //                                   |
  //                                   |
  //                                   v
  //                                   y (down)

  Eigen::MatrixXf C_ROS_T_L(4,4); // translation matrix lidar-camera
  C_ROS_T_L << C_ROS_R_L(0), C_ROS_R_L(3), C_ROS_R_L(6), C_ROS_t_L(0),
               C_ROS_R_L(1), C_ROS_R_L(4), C_ROS_R_L(7), C_ROS_t_L(1),
               C_ROS_R_L(2), C_ROS_R_L(5), C_ROS_R_L(8), C_ROS_t_L(2),
               0,        0,        0,        1;

  Eigen::MatrixXf C_T_C_ROS(4,4);
  C_T_C_ROS <<  0, -1,  0,  0,   // x = -Y
                0,  0, -1,  0,   // y = -Z
                1,  0,  0,  0,   // z = X
                0,  0,  0,  1;
  
  C_T_L = C_T_C_ROS * C_ROS_T_L;
}

void callback(const boost::shared_ptr<const sensor_msgs::PointCloud2>& in_pc2 , const ImageConstPtr& in_image)
{
  // From ROS image to OpenCV image
  cv_bridge::CvImagePtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvCopy(in_image, sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  // Copy the image
  cv_ptr->image.copyTo(orig_image);

  //Conversion from sensor_msgs::PointCloud2 to pcl::PointCloud<T>
  pcl::PCLPointCloud2 pcl_pc2;
  pcl_conversions::toPCL(*in_pc2, pcl_pc2);
  pcl::PointCloud<pcl::PointXYZ>::Ptr msg_pointCloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromPCLPointCloud2(pcl_pc2, *msg_pointCloud);
  ///

  ////// filter point cloud
  if (msg_pointCloud == NULL) return;

  PointCloud::Ptr cloud_in (new PointCloud);
  //PointCloud::Ptr cloud_out (new PointCloud);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_out_color(new pcl::PointCloud<pcl::PointXYZRGB>);
  cloud_out_color->header.frame_id = "velodyne";

  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*msg_pointCloud, *cloud_in, indices);

  // SWAN: range
  float min_range = 100000.f;
  float max_range = 0.f;
  pcl::PointXYZRGB point;
  for (int i = 0; i < (int) cloud_in->points.size(); i++)
  {
      // double distance = sqrt(cloud_in->points[i].x * cloud_in->points[i].x + cloud_in->points[i].y * cloud_in->points[i].y);
      // if(distance < minlen || distance > maxlen || cloud_in->points[i].x < 0)
      //     continue;
      // float ang = atan(cloud_in->points[i].x / cloud_in->points[i].y);
      // if(cloud_in->points[i].y < 0)
      //   ang = M_PI+ ang;
      // if (ang < min_FOV || ang > max_FOV)
      //     continue;

      point.x = cloud_in->points[i].x;
      point.y = cloud_in->points[i].y;
      point.z = cloud_in->points[i].z;
      // getColor(cloud_in->points[i].x, min_range, max_range, point.r, point.g, point.b);
      cloud_out_color->push_back(point);

      // Only consider the points in-front
      if (cloud_in->points[i].x <= 0) continue;
      if (cloud_in->points[i].x < min_range) min_range = cloud_in->points[i].x;
      if (cloud_in->points[i].x > max_range) max_range = cloud_in->points[i].x;
  }

  int n = (int) cloud_out_color->points.size(); // number of points
  Eigen::MatrixXf I_p(3, n);
  Eigen::MatrixXf L_p(4, n);
  Eigen::MatrixXf C_p(4, n);

  for (int i = 0; i < n; i++)
  {
      // L_p(0,i) = -cloud_out_color->points[i].y;
      // L_p(1,i) = -cloud_out_color->points[i].z;
      // L_p(2,i) =  cloud_out_color->points[i].x;

      L_p(0,i) = cloud_out_color->points[i].x;
      L_p(1,i) = cloud_out_color->points[i].y;
      L_p(2,i) = cloud_out_color->points[i].z;

      L_p(3,i) = 1.0;
  }

  // I_p = K * [C_R_L, C_t_L] * L_p
  // 3xn  3x3       3x4        4xn

  // I_p = [K, 0] * [C_R_L, C_t_L] * L_p
  //                [  0,     1  ]
  // 3xn     3x4          4x4        4xn

  // I_p = M * L_T_C * L_p
  // 3xn  3x4   4x4    4xn

  // I_p = M * C_p
  // 3xn  3x4  4xn

  // C_p = C_T_L * L_p
  C_p = C_T_L * L_p;

  I_p = M * C_p;

  float x, y, z;
  int u = 0;
  int v = 0;
  unsigned int cols = in_image->width;
  unsigned int rows = in_image->height;

  // float normalized_range;
  uint8_t R, G, B;
  for (int i = 0; i < n; i++)
  {
    // SWAN: consider forward LiDAR points only
    if (cloud_out_color->points[i].x <= 0) continue;

    // C_p
    x = I_p(0,i);
    y = I_p(1,i);
    z = I_p(2,i);

    // I_p
    u = (int)(x/z);
    v = (int)(y/z);

    if(u < 0.0 || u > cols || v < 0.0 || v > rows) continue;

    // Colorize the LiDAR points
    // cv::Mat_<cv::Vec3b> I = orig_image;
    // cloud_out_color->points[i].b = I(v, u)[0];
    // cloud_out_color->points[i].g = I(v, u)[1];
    // cloud_out_color->points[i].r = I(v, u)[2];
    cloud_out_color->points[i].b = orig_image.at<cv::Vec3b>(v, u)[0];
    cloud_out_color->points[i].g = orig_image.at<cv::Vec3b>(v, u)[1];
    cloud_out_color->points[i].r = orig_image.at<cv::Vec3b>(v, u)[2];
    // Project on the image

    // normalized_range = (C_p(2, i) - min_z) / max_z; // [0, 1]
    //setRainbowColor(normalized_range, R, G, B);
    //getColor(L_p(0, i), min_range, max_range, R, G, B);
 // int color_dis_x = (int)(255*((cloud_out->points[i].x)/maxlen));
    // int color_dis_x = (int)(255*((cloud_out->points[i].x)/10.0));
    // int color_dis_z = (int)(255*((cloud_out->points[i].x)/20.0));
    // if(color_dis_z > 255) color_dis_z = 255;
    
    getColor(cloud_out_color->points[i].x, min_range, max_range, R, G, B);

    //cv::circle(cv_ptr->image, cv::Point(u, v), 5, CV_RGB(255-color_dis_x,(int)(color_dis_z),color_dis_x),cv::FILLED);
    cv::circle(cv_ptr->image, cv::Point(u, v), 5, CV_RGB(R, G, B), cv::FILLED);
  }

   pcOnimg_pub.publish(cv_ptr->toImageMsg());
   pcl_conversions::toPCL(ros::Time::now(), cloud_out_color->header.stamp);
   pub.publish(cloud_out_color);
}

int main(int argc, char** argv)
{

  ros::init(argc, argv, "pontCloudOntImage");
  ros::NodeHandle nh;  

  /// Load Parameters

  // nh.getParam("/maxlen", maxlen);
  // nh.getParam("/minlen", minlen);
  // nh.getParam("/max_ang_FOV", max_FOV);
  // nh.getParam("/min_ang_FOV", min_FOV);
  nh.getParam("/pcTopic", pcTopic);
  nh.getParam("/imgTopic", imgTopic);

  XmlRpc::XmlRpcValue param;

  // Extrinsic parameters
  // L_T_C: transform points from camera coords to lidar coords
  Eigen::MatrixXf L_R_C_ROS(3,3);
  Eigen::MatrixXf L_t_C_ROS(3,1);

  nh.getParam("/matrix_file/rlc", param);
  L_R_C_ROS << (double)param[0], (double)param[1], (double)param[2],
               (double)param[3], (double)param[4], (double)param[5],
               (double)param[6], (double)param[7], (double)param[8];

  nh.getParam("/matrix_file/tlc", param);
  L_t_C_ROS << (double)param[0],
               (double)param[1],
               (double)param[2];

  nh.getParam("/matrix_file/camera_matrix", param);
  M  << (double)param[0], (double)param[1], (double)param[2],  (double)param[3],
        (double)param[4], (double)param[5], (double)param[6],  (double)param[7],
        (double)param[8], (double)param[9], (double)param[10], (double)param[11];

  nh.getParam("/matrix_file/distortion", param);
  D << (double)param[0], (double)param[1], (double)param[2],  (double)param[3], (double)param[4];

  // SWAN
  reviseTransformation(L_R_C_ROS, L_t_C_ROS);

  message_filters::Subscriber<PointCloud2> pc_sub(nh, pcTopic , 1);
  message_filters::Subscriber<Image> img_sub(nh, imgTopic, 1);

  typedef sync_policies::ApproximateTime<PointCloud2, Image> MySyncPolicy;
  const double time_tolerance = 10; // seconds
  Synchronizer<MySyncPolicy> sync(MySyncPolicy(time_tolerance), pc_sub, img_sub);
  sync.registerCallback(boost::bind(&callback, _1, _2));
  pcOnimg_pub = nh.advertise<sensor_msgs::Image>("/pcOnImage_image", 1);

  //pub = nh.advertise<PointCloud> ("/points2", 1);
  pub = nh.advertise<ColorPointCloud> ("/points2", 1);

  ros::spin();
  //return 0;
}
