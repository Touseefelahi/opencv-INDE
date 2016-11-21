#include "opencv2/opencv.hpp"
#include "opencv2/core/ocl.hpp"
#include <iostream>
#include <opencv2\core\cuda.hpp>

using namespace cv;
using namespace std;
using namespace cuda;

int main(int argc, char** argv)
{
	ocl::setUseOpenCL(true);
	UMat gpuFrame;
	UMat gpuBW;
	//GpuMat gpuFrame;
	UMat gpuBlur;
	UMat gpuEdges;
	VideoCapture cap("my movie.mp4"); // open the default camera
	if (!cap.isOpened()) 	return -1;
//	cap("my movie.mp4");
	namedWindow("edges", 1);
	for (;;)
	{
		double t = (double)cv::getTickCount();
		cap >> gpuFrame; // get a new frame from camera
		cvtColor(gpuFrame, gpuBW, COLOR_BGR2GRAY);
		GaussianBlur(gpuBW, gpuBlur, Size(1, 1), 1.5, 1.5);
		int a = 0; while (a < 1){ Canny(gpuBlur, gpuEdges, 0, 30, 3); a++; }

		imshow("edges", gpuEdges);
		if (waitKey(5) >= 0) break;
		t = (double)cv::getTickCount() - t;
		printf("detection time = %gms\n", t*1000. / cv::getTickFrequency());
	}
	// the camera will be deinitialized automatically in VideoCapture destructor
	return 0;
}