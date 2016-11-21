#include "opencv2/opencv.hpp"
#include <opencv2\core\ocl.hpp>

using namespace cv;
using namespace ocl;
int main(int argc, char** argv)
{
	UMat img, gray;
	static int tt = 0;
	imread("graf1.jpg", 1).copyTo(img);
	double t = (double)cv::getTickCount();
	while (tt < 100){
		cvtColor(img, gray, COLOR_BGR2GRAY);
		GaussianBlur(gray, gray, Size(7, 7), 1.5);
		Canny(gray, gray, 0, 50);
		
		tt++;
		imshow("edges", gray);
	}
	t = (double)cv::getTickCount() - t;

	printf("detection time = %gms\n", (t*1000. / cv::getTickFrequency()));
	waitKey(500);
	return 0;
}