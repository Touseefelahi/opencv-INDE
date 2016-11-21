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
bool Newframe = true, Newobj = false, debug = false,match=true;
Point point1, point2;
int drag = 0,xx,yy,lx,ly;
float s;
Rect rect;
float factor = 1; //pixel to distance
Mat img, roiImg; 
int select_flag = 0;
bool go_fast = false;
int64 initialtick = cvGetTickCount();
Mat mytemplate;

static int64  currentTime, lastTime = cvGetTickCount();
static float freq,sec = 0, speed = 0, timee = 0;

Mat frame;
vector<Point2f> bb;
const double akaze_thresh = 2e-4;
const double ransac_thresh = 3.1f; 
const double nn_match_ratio = 0.8f; 
const int bb_min_inliers = 20; 
class Tracker
{
public:
	Tracker(Ptr<Feature2D> _detector, Ptr<DescriptorMatcher> _matcher) :
		detector(_detector),
		matcher(_matcher)
	{}

	void setFirstFrame(const Mat frame, vector<Point2f> bb, string title, Stats& stats);

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
	first_frame = frame.clone();
	//Mat new1 = frame.clone();
	Mat mask = frame.clone(); mask = imread("obj.jpg");
//	Mat mask(frame.size(), CV_16UC1, Scalar(1));
	int wi = bb[2].x - bb[0].x;
	int hi = bb[1].y - bb[0].y;
	Mat roi(first_frame, cv::Rect(bb[0].x, bb[0].y,wi-1, hi-1));
	cv::Rect roi2(cv::Point(bb[0].x, bb[0].y), roi.size());
	roi.copyTo(mask(roi2));
	/*imshow("roi", mask);
	imshow("mask", roi);*/
	

//	detector->detectAndCompute(first_frame, noArray(), first_kp, first_desc);
	detector->detectAndCompute(mask, noArray(), first_kp, first_desc);
	/*detector->detect(mask, first_kp, first_desc);
	detector->compute(mask, first_kp, first_desc);*/
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
		//cout << homography << endl;
	}

	if (matched1.size() < 4 || homography.empty()) {
		Mat res;
		if (match == true|| debug==true) cout << "No. of matches <4"  << endl;
		hconcat(first_frame, frame, res);
		stats.inliers = 0;
		stats.ratio = 0;
		if (debug == true)
			return res;
		else
			return frame;
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
	if (debug == true || match == true) cout << "No. of matches" << inliers2.size() << endl;
	vector<Point2f> new_bb;
	perspectiveTransform(object_bb, new_bb, homography);
	Mat frame_with_bb = frame.clone();
	if (stats.inliers >= bb_min_inliers) {
		xx = (new_bb[0].y + new_bb[2].y) / 2;
		yy = (new_bb[1].x + new_bb[3].x) / 2;
		drawBoundingBox(frame_with_bb, new_bb);
		s = pow((xx - lx), 2) + pow((yy - ly), 2);
		s = sqrt(s);
		currentTime = cvGetTickCount();
		timee = (currentTime - lastTime)/(1e6);
		timee = timee / cvGetTickFrequency();
		speed = s / timee; speed = speed*factor;
	//	float totaltime = (currentTime - initialtick) / (1e6 * cvGetTickFrequency());
	//	speed = floorf(speed * 100) / 100; timee = floorf(timee * 100) / 100;
	//	cout << xx << "," << yy << " To" << lx << "," << ly <<" Distance and Time="<<s<<" ,"<<timee<<"sec"<<" pixels"<<" speed= "<<speed<< "pixels/seconds"<< endl;
	//	cout << "speed = " << speed << endl;
	//	cout << "Time = " << timee<<" Total Time "<<totaltime << endl;
		
		stringstream str1;
		str1 << "Speed " << speed<<" pixel/sec";
		putText(frame_with_bb, str1.str(), Point(10, 450), FONT_HERSHEY_PLAIN, 1.5, cvScalar(0, 0, 190), 2, CV_AA);
	//	if (new_bb[0].y>0 && new_bb[1].y>0&&new_bb[1].x >0&& new_bb[2].x>0)
		{
		//	line(frame_with_bb, Point(new_bb[0].x, new_bb[0].y), Point(new_bb[2].x, new_bb[2].y), Scalar(0, 0, 255), 2, 8, 0);
		//	line(frame_with_bb, Point(new_bb[1].x, new_bb[1].y), Point(new_bb[3].x, new_bb[3].y), Scalar(0, 0, 255), 2, 8, 0);
			circle(frame_with_bb, Point(yy, xx), 2, (255, 0, 0), 2, 8, 0);
		}
		lastTime = currentTime;
		lx = xx;
		ly = yy;

	}
	Mat res;
	drawMatches(first_frame, inliers1, frame_with_bb, inliers2,
		inlier_matches, res,
		Scalar(255,255,255), Scalar(0, 0, 255));
	if (debug==true)
	return res;
	else
	return frame_with_bb;
	
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
	freq = cvGetTickFrequency();
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
	//Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-L1");
	Tracker akaze_tracker(akaze, matcher);
	
	
//	cout << "New object? (y/n)" << endl;
	video_in >> frame; Newframe = true; Newobj = false;
	while (1) {
	if (Newframe == true)
	{
		
		while (select_flag==0)
		{
			Mat frame2,frame3;
			video_in >> frame;
		//	cvtColor(frame,frame, CV_BGR2GRAY);
		//	equalizeHist(frame, frame);
			putText(frame, "Select ROI by Mouse", Point(0, 20), FONT_HERSHEY_PLAIN, 1.5, Scalar::all(0), 2);
		//	drawBoundingBox(frame, bb);
			imshow("capture", frame);
//			imshow("bw", frame2);
//			imshow("equlized", frame3);
			cvSetMouseCallback("capture", mouseHandler, NULL);
			waitKey(30);
		}
		cout << "boundingBox 0=" << bb[0] << endl;
		cout << "boundingBox 1=" << bb[1] << endl;
		cout << "boundingBox 2=" << bb[2] << endl;
		cout << "boundingBox 3=" << bb[3] << endl;
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
	
		switch (waitKey(1)){
		case 27: //'esc'
			return 0;
		case 113: //'q' 
			return 0;
		case 116: //'t'
			Newframe = !Newframe;
			select_flag = 0;
			if (Newframe == true) cout << "Capturing New Frame" << endl;
			destroyWindow("output");
			break;
		case 112: //'p'
				cout << "Paused" << endl;
			while (waitKey(10) != 'p') {
			}
		
			cout << "Resumed" << endl;
			break;
		case 100: //'d'
			debug =!debug;
		
			if (debug == false) { cout << "Debug Mode off" << endl; break; }
		 cout << "Debug Mode on" << endl;
			break;
		case 109: //'m'
			match = !match;
			if (match==true||debug==true) cout << "show matches=on" << endl;
			else cout << "show matches=off" << endl;
			break;
		}
	}
//	printStatistics("AKAZE", akaze_stats);
	return 0;
}