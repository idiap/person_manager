#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

static const std::string OPENCV_WINDOW = "Image window";

class ImageViewer
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher image_pub_;

public:
  ImageViewer(const std::string& name):it_(nh_)
  {
    image_sub_ = it_.subscribe(name, 1, &ImageViewer::callback, this);
    cv::namedWindow(OPENCV_WINDOW);
  }

  ~ImageViewer()
  {
    cv::destroyWindow(OPENCV_WINDOW);
  }

  void callback(const sensor_msgs::ImageConstPtr& msg)
  {
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg,
                                   sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    // Update GUI Window
    cv::imshow(OPENCV_WINDOW, cv_ptr->image);
    cv::waitKey(3);

  }
};

int main(int argc, char** argv)
{
  std::string name = (argc>1) ? argv[1] : "/camera/rgb/image_raw";
  ros::init(argc, argv, "image_viewer");
  ImageViewer iv(name);
  ros::spin();
  return 0;
}
