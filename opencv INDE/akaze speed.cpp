#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <conio.h>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <iomanip>

#include "stats.h" // Stats structure definition
#include "utils.h"
//Name spaces used
const double akaze_thresh = 2e-4;
const double ransac_thresh = 3.1f;
const double nn_match_ratio = 0.9f;
const int bb_min_inliers = 10;
const int stats_update_period = 10;
using namespace cv;
using namespace std;
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

	Ptr<Feature2D> detector;
	Ptr<DescriptorMatcher> matcher;
	Mat first_frame, first_desc;
	vector<KeyPoint> first_kp;
	vector<Point2f> object_bb;
};
int main()
{
	float thresholdMatchingNN = 0.65;
	int gd = 5;
	int i = 41;//size
	//turn performance analysis functions on if testing = true
	double t; //timing variable
	VideoCapture cap(0);
	//	cap.open("tank1.mp4");     Mat object = imread ("missile.jpg", CV_LOAD_IMAGE_GRAYSCALE);   char* objj="Missile";  thresholdMatchingNN=0.75;  gd=7;
	//	cap.open("airplane.mp4");  Mat object = imread ("airplane.jpg", CV_LOAD_IMAGE_GRAYSCALE); char*objj="Airplane";
	Mat object = imread("airplane.jpg", CV_LOAD_IMAGE_GRAYSCALE); char*objj = "Airplane";
	//	Mat object = imread ("missile.jpg", CV_LOAD_IMAGE_GRAYSCALE);   char* objj="Missile";
	//cap.set(CV_CAP_PROP_FRAME_WIDTH,240);
	//cap.set(CV_CAP_PROP_FRAME_HEIGHT,160);
	//load training image 

	int tag = 0;

	if (!object.data){
		cout << "Can't open image";
		return -1;
	}

	namedWindow("Good Matches", CV_WINDOW_AUTOSIZE);

	//SURF Detector, and descriptor parameters
	vector<KeyPoint> kpObject, kpImage;
	Mat desObject, desImage;
	Stats stats, akaze_stats;
	//Ptr<Feature2D> detector;
	//Ptr<DescriptorMatcher> matcher;
	Ptr<AKAZE> akaze = AKAZE::create();
	akaze->setThreshold(akaze_thresh);
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	Tracker aka(akaze, matcher);
	aka.detector->detectAndCompute(object, noArray(), kpObject,desObject);
	stats.keypoints = (int)kpObject.size();
	
	//Initialize video and display window
	//	VideoCapture cap(0);  //camera 1 is webcam
	char mat[20], mat2[20], mat3[3];
	if (!cap.isOpened()) return -1;

	//Object corner points for plotting box
	vector<Point2f> obj_corners(4);
	obj_corners[0] = cvPoint(0, 0);
	obj_corners[1] = cvPoint(object.cols, 0);
	obj_corners[2] = cvPoint(object.cols, object.rows);
	obj_corners[3] = cvPoint(0, object.rows);

	//video loop
	char escapeKey = 'k';
	double frameCount = 0;

	unsigned int thresholdGoodMatches = 4;
	unsigned int thresholdGoodMatchesV[] = { 4, 5, 6, 7, 8, 9, 10 };

	for (int j = 0; j<7; j++){
		thresholdGoodMatches = thresholdGoodMatchesV[j];
		//thresholdGoodMatches=8;
		cout << thresholdGoodMatches << endl;

		t = (double)getTickCount();


		while (escapeKey != 'q')
		{
			frameCount++;
			Mat frame;
			Mat image;
			cap >> frame;

		
			if (!frame.data){
				cout << "Can't open image";
				return 0;
			}
		sub:
			cvtColor(frame, image, CV_RGB2GRAY);
			rectangle(image, Point(0, 320), Point(200, 360), Scalar(0, 0, 255), -1, 8, 0);
			Mat des_image, img_matches, H;
			vector<KeyPoint> kp_image,kp;
			vector<vector<DMatch > > matches;
			vector<DMatch > good_matches;
			vector<Point2f> obj;
			vector<Point2f> scene, x, y, z;
			vector<Point2f> scene_corners(7);
			aka.detector->detectAndCompute(image, noArray(), kpImage, des_image);
			stats.keypoints = (int)kp.size();
			matcher->knnMatch(desObject, des_image, matches, 2);
			
			int t = (int)matches.size();
			for (int i = 0; i < min(des_image.rows - 1, (int)matches.size()); i++) //THIS LOOP IS SENSITIVE TO SEGFAULTS
			{
				if ((matches[i][0].distance < thresholdMatchingNN*(matches[i][1].distance)) && ((int)matches[i].size() <= 2 && (int)matches[i].size()>0))
				{
					good_matches.push_back(matches[i][0]);
				}
			}

			//if (good_matches.size()<1)
			//	good_matches.resize(0,cv::DMatch);

			//Draw only "good" matches
			drawMatches(object, kpObject, image, kp_image, good_matches, img_matches, Scalar(150, 0, 0), Scalar(150, 0, 0), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
			//	drawMatches( object, kpObject, image, kp_image, good_matches, img_matches, Scalar(150,0,0), Scalar::all(-1), vector<char>(), DrawMatchesFlags:: DRAW_RICH_KEYPOINTS );
			//img_matches=image;
			//cout << "Good matches= "<<good_matches.size()<<endl;
			if (good_matches.size() >= gd)
			{
				char ob[20];
				sprintf_s(ob, "Object=%s", objj);

				//Display that the object is found
				putText(img_matches, ob, cvPoint(10, 120), FONT_HERSHEY_COMPLEX_SMALL, .8, cvScalar(0, 0, 250), 1, CV_AA);

				for (unsigned int i = 0; i < good_matches.size(); i++)
				{
					//Get the keypoints from the good matches
					obj.push_back(kpObject[good_matches[i].queryIdx].pt);
					scene.push_back(kp_image[good_matches[i].trainIdx].pt);
				}

				H = findHomography(obj, scene, CV_RANSAC);
				//	cout <<"Homography="<< H << endl;
				////	getch();
				perspectiveTransform(obj_corners, scene_corners, H);

				//	int x=180+(scene_corners[0].x+scene_corners[1].x)/2;
				//	int y=10+(scene_corners[0].y+scene_corners[3].y)/2;
				////	Draw lines between the corners (the mapped object in the scene image )
				line(img_matches, scene_corners[0] + Point2f(object.cols, 0), scene_corners[1] + Point2f(object.cols, 0), Scalar(0, 255, 0), 2);
				line(img_matches, scene_corners[1] + Point2f(object.cols, 0), scene_corners[2] + Point2f(object.cols, 0), Scalar(0, 255, 0), 2);
				line(img_matches, scene_corners[2] + Point2f(object.cols, 0), scene_corners[3] + Point2f(object.cols, 0), Scalar(0, 255, 0), 2);
				line(img_matches, scene_corners[3] + Point2f(object.cols, 0), scene_corners[0] + Point2f(object.cols, 0), Scalar(0, 255, 0), 2);
				/*	line(img_matches,Point(x,y),Point(x,y-25),Scalar(0,255,0),2);
				//	line(img_matches,Point(x,y),Point(x,y+25),Scalar(0,255,0),2);
				//	line(img_matches,Point(x,y),Point(x-25,y),Scalar(0,255,0),2);
				//	line(img_matches,Point(x,y),Point(x+25,y),Scalar(0,255,0),2);*/
			}
			else
			{
				putText(img_matches, "", cvPoint(10, 50), FONT_HERSHEY_COMPLEX_SMALL, 3, cvScalar(0, 0, 250), 1, CV_AA);
				//	object= imread ("mizile.jpg", CV_LOAD_IMAGE_GRAYSCALE); goto sub;
			}


			static int64 currentTime, lastTime = cvGetTickCount();
			static int fpsCounter = 0, fps, sec = 0;
			currentTime = cvGetTickCount();
			++fpsCounter;

			// If 1 second has passed since the last FPS estimation, update the fps
			if (currentTime - lastTime > 1e6 * cvGetTickFrequency()) {
				fps = fpsCounter;
				lastTime = currentTime;
				fpsCounter = 0;
				sec++;
			}

			sprintf_s(mat, "No. of Matches=%d", good_matches.size());
			putText(img_matches, mat, cvPoint(10, 140), FONT_HERSHEY_COMPLEX_SMALL, .7, cvScalar(0, 255, 250), 1, CV_AA);
			sprintf_s(mat2, "FPS= %d", fps);
			//	putText(img_matches, mat2, cvPoint(10,160),FONT_HERSHEY_COMPLEX_SMALL, .7, cvScalar(0,255,250), 1, CV_AA);
			//	sprintf(mat3,"Seconds=%d",sec);	putText(img_matches, mat3, cvPoint(10,180),FONT_HERSHEY_COMPLEX_SMALL, .7, cvScalar(0,255,250), 1, CV_AA);

			//Show detected matches
			imshow("Good Matches", img_matches);
			escapeKey = cvWaitKey(10);

			//	if(tag%2==0)				{object= imread ("mizile4.jpg", CV_LOAD_IMAGE_GRAYSCALE); goto sub;}

			//	if (tag%2!=0)						{ object= imread ("mizile.jpg", CV_LOAD_IMAGE_GRAYSCALE); goto sub; }
			//imwrite("C:/School/Image Processing/bookIP3.jpg", img_matches);

			//  if(frameCount>10) escapeKey='q';


		}
		//sprintf(mat3,"Seconds=%d",5);
		//average frames per second
		if (true)
		{
			t = ((double)getTickCount() - t) / getTickFrequency();
			cout << t << " " << frameCount / t << endl;
			cvWaitKey(0);
		}

		frameCount = 0;
		escapeKey = 'a';
	}

	//Release camera and exit
	cap.release();
	return 0;
	exit(0);
}