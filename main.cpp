#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

#define MEAN 0
#define MEDIAN 1
#define GAUSSIAN 2
#define BILATERAL 3

int main(int argc, char *argv[])
{
    namedWindow("img", WINDOW_KEEPRATIO);
    namedWindow("img2", WINDOW_KEEPRATIO);

    Mat img = imread("img/lena.png", IMREAD_COLOR);
    Mat img2;
    int type = GAUSSIAN;
    int ksize = 3, sigma = 3;

    createTrackbar("filter type", "img2", &type, 3, 0, 0);
    createTrackbar("ksize", "img2", &ksize, 20, 0, 0);
    createTrackbar("sigma", "img2", &sigma, 100, 0, 0);

    for(;;) {
        switch (type) {
        case MEAN:
            blur(img, img2, Size((ksize+1)*2 - 1, (ksize+1)*2 - 1));
            break;
        case MEDIAN:
            medianBlur(img, img2, (ksize+1)*2 - 1);
            break;
        case GAUSSIAN:
            GaussianBlur(img, img2, Size((ksize+1)*2 - 1, (ksize+1)*2 - 1), sigma);
            break;
        case BILATERAL:
            bilateralFilter(img, img2, 9, sigma, sigma);
            break;
        default:
            break;
        }

        imshow("img", img);
        imshow("img2", img2);

        if ((char)waitKey(5) == 'q') break;
    }

    return 0;
}
