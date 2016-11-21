#include <opencv/cv.h>
#include <iostream>
#include "opencv2/highgui/highgui.hpp"
using namespace cv;
int main(int argc, char *argv[])
{
	
//  cv::VideoCapture cap("http://admin:4321@10.1.11.173:4321/home/monitoring.cgi?/GetOneShot?image_size=320x240&frame_count=no_limit&type=.mjpg");
//	cv::VideoCapture cap("http://admin:4321@10.1.11.190:4321/cgi-bin/video.cgi?msubmenu=mjpg");
	cv::VideoCapture camout("rtsp://admin:4321@192.168.10.69:554/onvif/profile2/media.smp");
	cv::VideoCapture camin("rtsp://admin:4321@192.168.10.88:554/onvif/profile2/media.smp");

//	cv::VideoCapture cap("rtsp://10.1.11.191/Streaming/Channels/1 ");
//	cv::VideoCapture cap("http://admin:12345@10.1.11.191/doc/page/main.asp");
  //  bool a =VideoCapture::grab;// nn("rtsp://admin:12345@10.1.11.191:554/Streaming/Channels/1 ");
	cv::Mat frame_out;
	cv::Mat frame_in;
	cv::namedWindow("camout",CV_WINDOW_FREERATIO);
	cvMoveWindow("camout", 100, 0);
	cv::namedWindow("camin", CV_WINDOW_FREERATIO);
	cvMoveWindow("camin", 600, 0);

	int a = 0;
	while (camout.isOpened())
	{
		camout >> frame_out;
		camin >> frame_in;

		//if (frame_out.empty()) break;

		cv::imshow("camout", frame_out);
		
		cv::imshow("camin", frame_in);

		if (int c=cv::waitKey(30) == 27) break;
	}

	return 0;
}