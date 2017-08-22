#include "examples.h"

Mat scaleImage2_uchar(Mat &src) {
    Mat tmp = src.clone();
    if (src.type() != CV_32F)
        tmp.convertTo(tmp, CV_32F);
    normalize(tmp, tmp, 1, 0, NORM_MINMAX);
    tmp = 255 * tmp;
    tmp.convertTo(tmp, CV_8U, 1, 0);
    return tmp;
}

Mat cvtImg2Colormap(const Mat &src, int colormap) {
    Mat output = src.clone();
    output = scaleImage2_uchar(output);
    applyColorMap(output, output, colormap);
    return output;
}

int example1() {

    namedWindow("baboon", WINDOW_KEEPRATIO);
    namedWindow("b", WINDOW_KEEPRATIO);
    namedWindow("g", WINDOW_KEEPRATIO);
    namedWindow("r", WINDOW_KEEPRATIO);

    Mat baboon = imread("img/baboon.png", IMREAD_COLOR);
    std::vector<Mat> bgr;
    split(baboon, bgr);

    imshow("baboon", baboon);
    imshow("b", cvtImg2Colormap(bgr[0], COLORMAP_JET));
    imshow("g", cvtImg2Colormap(bgr[1], COLORMAP_JET));
    imshow("r", cvtImg2Colormap(bgr[2], COLORMAP_JET));

    for (;;)
        if ((char)waitKey(1) == 'q') break;

    return 0;
}

int example2() {

    namedWindow("img", WINDOW_KEEPRATIO);
    namedWindow("b", WINDOW_KEEPRATIO);
    namedWindow("g", WINDOW_KEEPRATIO);
    namedWindow("r", WINDOW_KEEPRATIO);

    Mat img = imread("img/rgbcube_kBKG.png", IMREAD_COLOR);
    std::vector<Mat> bgr;
    img = Scalar(255, 255, 255) - img;
    split(img, bgr);

    imshow("img", img);
    imshow("b", bgr[0]);
    imshow("g", bgr[1]);
    imshow("r", bgr[2]);

    for (;;)
        if ((char)waitKey(1) == 'q') break;

    return 0;
}

int example3() {

    namedWindow("img", WINDOW_KEEPRATIO);
    namedWindow("y", WINDOW_KEEPRATIO);
    namedWindow("r", WINDOW_KEEPRATIO);
    namedWindow("b", WINDOW_KEEPRATIO);

    Mat img = imread("img/rgbcube_kBKG.png", IMREAD_COLOR);
    Mat img2;
    std::vector<Mat> yrb;
    cvtColor(img, img2, CV_BGR2YCrCb);
    split(img, yrb);

    imshow("img", img);
    imshow("y", yrb[0]);
    imshow("r", yrb[1]);
    imshow("b", yrb[2]);

    for (;;)
        if ((char)waitKey(1) == 'q') break;

    return 0;
}

int main(int argc, char *argv[])
{
    return example3();
}
