#ifndef UTILS_H
#define UTILS_H

#include "common.h"

class Utils {
public:
    Utils();

    Mat computeHistogram1C (const Mat &src);
    Mat computeHistogram3C (const Mat &src);
    Mat scaleImage2_uchar(Mat &src);
    Mat createCosineImg(const int &rows, const int cols, const float &freq, const float &theta);
    Mat createWhiteDisk(const int &rows, const int &cols, const int &cx, const int &cy, const int &radius);
    Mat fftshift(const Mat &src);
    Mat cvtImg2Colormap(const Mat &src, int colormap);
};

#endif // UTILS_H
