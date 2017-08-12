#ifndef EXAMPLES_H
#define EXAMPLES_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

#define MEAN 0
#define MEDIAN 1
#define GAUSSIAN 2
#define BILATERAL 3

class Examples
{
public:
    Examples();

    int useWebcam();
    int splitImage();
    int useThreshold();
    int useGaussianNoise();
    int useWhiteNoise();
    int convertColorType();
    int negativeImage();
    int channelSum();
    int matrixCreation();
    int imagesDiff();
    int bitwiseOps();
    int cutImage();
    int gradient();
    int powerTransform();
    int logTransform();
    int linearRangeTransform();
    int bitLayerSplit();
    int tresholdTypes();
    int equalizeHistogram();
    int equalizeHistogramRange();
    int blurVerticalHorizontal();
    int medianBlurFilter();
    int fourBlurTypes();
    int sobelFilter();
    int easySobelFilter();
    int easySobelFilter2();
    int laplacianFilter();
    int easyLaplacianFilter();
    int dft();
    int dftMagnitude();
    int whiteDisk();
    int easyWhiteDisk();
    int lowPassFilterDFT();
    int highPassFilterDFT();
    int cosineImage();
    int cosineNoise();


    Mat createCosineImg(const int &rows, const int cols, const float &freq, const float &theta);
    Mat createWhiteDisk(const int &rows, const int &cols, const int &cx, const int &cy, const int &radius);
    Mat fftshift(const Mat &src);
    Mat scaleImage2_uchar(Mat &src);
    Mat computeHistogram1C (const Mat &src);
    Mat computeHistogram3C (const Mat &src);
    int piecewiseLinearTransform();
};

#endif // EXAMPLES_H
