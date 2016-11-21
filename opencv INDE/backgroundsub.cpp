#include <opencv2\opencv.hpp>
#include <vector>


using namespace cv;
using namespace std;
//http://admin:4321@10.1.11.190:4321/home/monitoring.cgi
int main(int argc, char *argv[])
{
	Ptr<BackgroundSubtractorMOG2> bg = createBackgroundSubtractorMOG2(500, 20, false);
//	VideoCapture cap(0);
VideoCapture cap("rtsp://admin:4321@10.1.11.190:554/onvif/profile2/media.smp");
	Mat3b frame;
	Mat1b fmask;
	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	Mat kernel1 = getStructuringElement(MORPH_RECT, Size(5, 5));
	for (;;)
	{
		// Capture frame
		cap >> frame;
		/*HOGDescriptor abc;
		cv::HOGDescriptor::getDefaultPeopleDetector;
		cv::HOGDescriptor::getDaimlerPeopleDetector*/
		// Background subtraction
		bg->apply(frame, fmask, -1);

		// Clean foreground from noise
		morphologyEx(fmask, fmask, MORPH_OPEN, kernel);
	//	morphologyEx(fmask, fmask, MORPH_GRADIENT, kernel);
	//	morphologyEx(fmask, fmask, MORPH_DILATE, kernel1);
		// Find contours
		vector<vector<Point>> contours;
		findContours(fmask.clone(), contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

		if (!contours.empty())
		{
			// Get largest contour
			int idx_largest_contour = -1;
			double area_largest_contour = 0.0;

			for (int i = 0; i < contours.size(); ++i)
			{
				double area = contourArea(contours[i]);
				if (area_largest_contour < area)
				{
					area_largest_contour = area;
					idx_largest_contour = i;
				}
			}

			if (area_largest_contour > 200)
			{
				// Draw
				Rect roi = boundingRect(contours[idx_largest_contour]);
				drawContours(frame, contours, idx_largest_contour, Scalar(0, 0, 255));
				rectangle(frame, roi, Scalar(0, 255, 0));
			}
		}

		imshow("frame", frame);
		imshow("mask", fmask);
		if (cv::waitKey(1) >= 0) break;
	}
	return 0;
}