#include <opencv2/features2d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <iomanip>

#include "stats.h" // Stats structure definition
#include "utils.h" // Drawing and printing functions

using namespace std;
using namespace cv;
VideoCapture video_in(0);
bool Newframe = true, Newobj = false;
Point point1, point2; /* vertical points of the bounding box */
int drag = 0;
Rect rect; /* bounding box */
Mat img, roiImg; /* roiImg - the part of the image in the bounding box */
int select_flag = 0;
bool go_fast = false;

Mat mytemplate;

Mat frame;
vector<Point2f> bb;
const double akaze_thresh = 3e-4; // AKAZE detection threshold set to locate about 1000 keypoints
const double ransac_thresh = 2.5f; // RANSAC inlier threshold
const double nn_match_ratio = 0.9f; // Nearest-neighbour matching ratio
const int bb_min_inliers = 20; // Minimal number of inliers to draw bounding box
const int stats_update_period = 1; // On-screen statistics are updated every 10 frames

class Tracker
{
public:
	Tracker(Ptr<Feature2D> _detector, Ptr<DescriptorMatcher> _matcher) :
		detector(_detector),
		matcher(_matcher)
	{}

	void setFirstFrame(const Mat frame, vector<Point2f> bb, string title, Stats& stats);
	void setFirstFrame2(const Mat frame, Rect bb, string title, Stats& stats);
	Mat process(const Mat frame, Stats& stats);
	Ptr<Feature2D> getDetector() {
		return detector;
	}
protected:
	Ptr<Feature2D> detector;
	Ptr<DescriptorMatcher> matcher;
	Mat first_frame, first_desc;
	vector<KeyPoint> first_kp;
	vector<Point2f> object_bb;
};

void Tracker::setFirstFrame(const Mat frame, vector<Point2f> bb, string title, Stats& stats)
{
	Mat mask = Mat::zeros(frame.size(), CV_8U);  // type of mask is CV_8U
	Mat roi(mask, cv::Rect(10, 10, 100, 100));
	first_frame = frame.clone();
//	detector->detectAndCompute(first_frame, noArray(), first_kp, first_desc);
	detector->detectAndCompute(first_frame, noArray(), first_kp, first_desc);
	stats.keypoints = (int)first_kp.size();
	drawBoundingBox(first_frame, bb);
//	putText(first_frame, title, Point(0, 60), FONT_HERSHEY_PLAIN, 5, Scalar::all(0), 4);
	object_bb = bb;
}

Mat Tracker::process(const Mat frame, Stats& stats)
{
	vector<KeyPoint> kp;
	Mat desc;
	detector->detectAndCompute(frame, noArray(), kp, desc);
	stats.keypoints = (int)kp.size();

	vector< vector<DMatch> > matches;
	vector<KeyPoint> matched1, matched2;
	matcher->knnMatch(first_desc, desc, matches, 2);
	for (unsigned i = 0; i < matches.size(); i++) {
		if (matches[i][0].distance < nn_match_ratio * matches[i][1].distance) {
			matched1.push_back(first_kp[matches[i][0].queryIdx]);
			matched2.push_back(kp[matches[i][0].trainIdx]);
		}
	}
	stats.matches = (int)matched1.size();

	Mat inlier_mask, homography;
	vector<KeyPoint> inliers1, inliers2;
	vector<DMatch> inlier_matches;
	if (matched1.size() >= 4) {
		homography = findHomography(Points(matched1), Points(matched2),
			RANSAC, ransac_thresh, inlier_mask);
	}

	if (matched1.size() < 4 || homography.empty()) {
		Mat res;
		hconcat(first_frame, frame, res);
		stats.inliers = 0;
		stats.ratio = 0;
		return res;
	}
	for (unsigned i = 0; i < matched1.size(); i++) {
		if (inlier_mask.at<uchar>(i)) {
			int new_i = static_cast<int>(inliers1.size());
			inliers1.push_back(matched1[i]);
			inliers2.push_back(matched2[i]);
			inlier_matches.push_back(DMatch(new_i, new_i, 0));
		}
	}
	stats.inliers = (int)inliers1.size();
	stats.ratio = stats.inliers * 1.0 / stats.matches;

	vector<Point2f> new_bb;
	perspectiveTransform(object_bb, new_bb, homography);
	Mat frame_with_bb = frame.clone();
	if (stats.inliers >= bb_min_inliers) {
		drawBoundingBox(frame_with_bb, new_bb);
	}
	Mat res;
	drawMatches(first_frame, inliers1, frame_with_bb, inliers2,
		inlier_matches, res,
		Scalar(255,255,255), Scalar(255, 0, 0));
	return res;
}

void mouseHandler(int event, int x, int y, int flags, void *param)
{	
//	video_in >> frame;
	if (event == CV_EVENT_LBUTTONDOWN && !drag)
	{
		/// left button clicked. ROI selection begins
		point1 = Point(x, y);
		drag = 1;
		
	}

	if (event == CV_EVENT_MOUSEMOVE && drag)
	{
		/// mouse dragged. ROI being selected
		Mat img1 = frame.clone();
		point2 = Point(x, y);
		rectangle(frame, point1, point2, CV_RGB(255, 0, 0), 3, 8, 0);
		imshow("capture", frame);
	}

	if (event == CV_EVENT_LBUTTONUP && drag)
	{
		point2 = Point(x, y);
		rect = Rect(point1.x, point1.y, x - point1.x, y - point1.y);
		drag = 0;
		/*roiImg = frame(rect);
		roiImg.copyTo(mytemplate);*/
		//imshow("MOUSE roiImg", roiImg); waitKey(0);
	}

	if (event == CV_EVENT_LBUTTONUP)
	{
		/// ROI selected
		Point2d point3, point4;
	//	point2 = Point(x, y);
		point3 = Point(point1.x, point1.y+(point2.y - point1.y));
		point4 = Point(point1.x + point2.x - point1.x, point1.y);
		Newobj = true;
		select_flag = 1;
		drag = 0;
		destroyWindow("capture");
		bb[0] = point1;
		bb[1] = point3;
		bb[2] = point2;
		bb[3] = point4;
		
	}
	
}

int main(int argc, char **argv)
{
	
		//("akaze.mp4");
	if (!video_in.isOpened()) {	cerr << "Couldn't open " << argv[1] << endl;	return 1;	}
	
	
	
	FileStorage fs("akaze.xml", FileStorage::READ);
	if (fs["bounding_box"].empty()) {
		cerr << "Couldn't read bounding_box from " << argv[3] << endl;
		return 1;	}
	fs["bounding_box"] >>bb;
	Stats stats, akaze_stats, orb_stats;
	Ptr<AKAZE> akaze = AKAZE::create();
	akaze->setThreshold(akaze_thresh);
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	Tracker akaze_tracker(akaze, matcher);
	
	
//	cout << "New object? (y/n)" << endl;
	video_in >> frame; Newframe = true; Newobj = false;
	while (1) {
	if (Newframe == true)
	{
		
		while (select_flag==0)
		{
			video_in >> frame;
			putText(frame, "Select ROI by Mouse", Point(0, 20), FONT_HERSHEY_PLAIN, 1.5, Scalar::all(0), 2);
		//	drawBoundingBox(frame, bb);
			imshow("capture", frame);
			cvSetMouseCallback("capture", mouseHandler, NULL);
			waitKey(30);
		}
	//	akaze_tracker.setFirstFrame(frame, bb, "AKAZE", stats);
	/*	cout << "boundingBox 0=" << bb[0] << endl;
		cout << "boundingBox 1=" << bb[1] << endl;
		cout << "boundingBox 2=" << bb[2] << endl;
		cout << "boundingBox 3=" << bb[3] << endl;*/
		Newframe = !Newframe;
		cvDestroyWindow("capture");
	}
	if (Newobj==true)
	{
		Newobj = !Newobj;
		akaze_tracker.setFirstFrame(frame, bb, "", stats);
		cvDestroyWindow("capture");
	}

	//frame = imread("obj.jpg");
	Stats akaze_draw_stats;
	
	Mat akaze_res;
		video_in >> frame;
		akaze_res = akaze_tracker.process(frame, stats);
	//	drawStatistics(akaze_res, akaze_draw_stats);
		
		if (rect.width == 0 && rect.height == 0)
		cvSetMouseCallback("output", mouseHandler, NULL);
		imshow("output", akaze_res); 
		//cvWaitKey(33);
	
		switch (waitKey(10)){
		case 27: //'esc' key has been pressed, exit program.
			return 0;
		case 113: //'q' key has been pressed, exit program.
			return 0;
		case 116: //'t'
			Newframe = !Newframe;
			select_flag = 0;
			if (Newframe == true) cout << "Capturing New Frame" << endl;
			destroyWindow("output");
			break;
		case 117: //'u'
			Newobj =true;
			//if (Newframe == false) cout << "Capturing New Frame" << endl;
			break;
	
		}
	}
	printStatistics("AKAZE", akaze_stats);
	return 0;
}