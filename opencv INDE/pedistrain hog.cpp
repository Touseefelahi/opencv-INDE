#include <opencv2/opencv.hpp>
#include <opencv2\core\ocl.hpp>
using namespace cv;
using namespace ocl;

int main(int argc, char **argv)
{
	// std::string inputFileName = argv[1];
	
	setUseOpenCL(true);
	if (!haveOpenCL()){ return 0; }
	cv::VideoCapture capture("pedestrians.avi");
	//cv::cuda::GpuMat frame24;
	if (!capture.isOpened())
		return -1;

	// cv::VideoWriter outputVideo;
	int frameWidth = 320;
	int frameHeight = 240;

	cv::Size frameSize = cv::Size(frameWidth, frameHeight);
	//  outputVideo.open("new.avi", -1, capture.get(CV_CAP_PROP_FPS), frameSize, true);

	cv::HOGDescriptor hog;
	hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
	// hog.setSVMDetector(cv::HOGDescriptor::getDaimlerPeopleDetector());
	Mat frame;
	UMat frame2;
//	cv::ocl::Image2D frame2;
	double t2 = (double)cv::getTickCount();
	int fc = 0;
	while (1)
	{
		capture >> frame;
		frame.copyTo(frame2);
		
	//	frame2.upload(frame);
		if (frame2.empty())
			break;

		cv::resize(frame2, frame2, frameSize);

		std::vector<cv::Rect> found, found_filtered;
		double t = (double)cv::getTickCount();
		fc++;
		if (fc>1000)  hog.detectMultiScale(frame2, found, 0, cv::Size(8, 8), cv::Size(16, 16), 1.05, 2);
		t = (double)cv::getTickCount() - t;

		printf("detection time = %gms\n", (t*1000. / cv::getTickFrequency()));

		size_t i, j;
		for (i = 0; i < found.size(); i++)
		{
			cv::Rect r = found[i];
			for (j = 0; j < found.size(); j++)
			if (j != i && (r & found[j]) == r)
				break;
			if (j == found.size())
				found_filtered.push_back(r);
		}

		for (i = 0; i < found_filtered.size(); i++)
		{
			cv::Rect r = found_filtered[i];
			// the HOG detector returns slightly larger rectangles than the real objects.
			// so we slightly shrink the rectangles to get a nicer output.
			r.x += cvRound(r.width*0.1);
			r.width = cvRound(r.width*0.8);
			r.y += cvRound(r.height*0.07);
			r.height = cvRound(r.height*0.8);

			rectangle(frame2, r.tl(), r.br(), cv::Scalar(0, 255, 0), 3);
		}
		
		// outputVideo << frame;
		imshow("capture", frame2);

		if (cv::waitKey(1) >= 0)
			break;
	}
	
	double t3 = (double)cv::getTickCount() - t2;
	printf("Total Time = %gs\n", (t3*1. / cv::getTickFrequency()));
	cvvWaitKey();
	return 0;
}
