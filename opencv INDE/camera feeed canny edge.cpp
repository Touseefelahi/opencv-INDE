/*
This code implements simple opencv pipeline that demonstrates 
how to read, process and display image using OpenCV library
*/
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;

int main(int argc, const char** argv)
{
    // declare capture engine that can read images from camera or file
    VideoCapture cap;

    // if no cmd-line arguments other than the app name then camera flag is raised
    bool camera = (1==argc);

    if(camera)
        // call open(int) method to init capture engine to read from camera
        // In case of many cameras the index of camera can be passed as argument.
        cap.open(0);
    else
        // call open(char*) method to init capture engine to read images from file
        // the argument is file name that will be opened for reading
        // it can be name of video file or still image
        cap.open(argv[1]);
    
    // check that capture engine open source (camera or file) successfully
    if (!cap.isOpened())
    {
        printf("can not open %s\n",camera?"camera":argv[1]);
        printf("trying to open test.jpg\n");
        // in case of fail try to open simple test file to be able check pipeline working
        cap.open("test.jpg");
        if (!cap.isOpened())
        {
            printf("can not open test.jpg\n");
            return EXIT_FAILURE;
        }
    }

    // prepare for processing images
    // declare mat objects to store input, intermediate and output images
    UMat imgInp;    // to store input image
    UMat imgGray;   // to store intermediate grayscale image
    UMat imgOut;    // to store final processing result

    // this is main loop over all input frames
    for (;;)
    {
        // get next frame from input stream
        cap >> imgInp;

        // check read result
        // in case of reading from file the loop will be break after last frame is read and processed
        // in case of camera this condition is always false until something wrong with camera 
        if (imgInp.empty())
        {
            // wait until user press any key and the break the loop
            // we need to wait to ge
            waitKey(0);
            break;
        }

        // show the input image on the screen using opencv function
        // this call creates window named "Input" and draws imgInp inside the window
        imshow("Input", imgInp);

        // convert input image into intermediate grayscale image
        cvtColor(imgInp, imgGray, COLOR_BGR2GRAY);
        
        // run canny processing on grayscale image
        Canny(imgGray, imgOut, 50, 150);

        // show the result on the screen using opencv function
        // this call creates window named "Canny" and draw imgOut inside the window
        imshow("Canny", imgOut);

        // the waitKey function is called for 2 reasons
        // 1. detect when ESC key is pressed
        // 2. to allow "Input" and "Canny" windows to plumb messages. It allows user to manipulate with "Input" and "Canny" windows
        // 10ms param is passed to spend only 10ms inside the waitKey function and then go to further processing
        int key = waitKey(10);

        //exit if ESC is pressed
        if (key == 27)
            break;
    }

    return EXIT_SUCCESS;
}