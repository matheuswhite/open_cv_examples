#include "examples.h"

Mat createWhiteDisk(const int &rows, const int &cols, const int &cx, const int &cy, const int &radius) {
    Mat disk = Mat::zeros(rows, cols, CV_32F);

    for (int x = 0; x < disk.cols; ++x) {
        for (int y = 0; y < disk.rows; ++y) {
            float d = pow((x-cx) * (x-cx) + (y-cy) * (y-cy), 0.5);
            if (d <= radius)
            {
                disk.at<float>(y, x) = 1.0;
                //disk.at<float>(x, y) = 1 - d / radius;
            }
        }
    }

    return disk;
}

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
    split(img2, yrb);

    imshow("img", img);
    imshow("y", yrb[0]);
    imshow("r", yrb[1]);
    imshow("b", yrb[2]);

    for (;;)
        if ((char)waitKey(1) == 'q') break;

    return 0;
}

int example4() {

    namedWindow("img", WINDOW_KEEPRATIO);
    namedWindow("h", WINDOW_KEEPRATIO);
    namedWindow("s", WINDOW_KEEPRATIO);
    namedWindow("v", WINDOW_KEEPRATIO);

    Mat img = imread("img/baboon.png", IMREAD_COLOR);
    Mat img2;
    std::vector<Mat> hsv;
    cvtColor(img, img2, CV_BGR2HSV);
    split(img2, hsv);

    imshow("img", img);
    imshow("h", hsv[0]);
    imshow("s", hsv[1]);
    imshow("v", hsv[2]);

    for (;;)
        if ((char)waitKey(1) == 'q') break;

    return 0;
}

int example5() {

    namedWindow("img", WINDOW_KEEPRATIO);
    namedWindow("h", WINDOW_KEEPRATIO);
    namedWindow("s", WINDOW_KEEPRATIO);
    namedWindow("v", WINDOW_KEEPRATIO);

    Mat img = imread("img/chips.png", IMREAD_COLOR);
    Mat img2;
    std::vector<Mat> hsv;
    cvtColor(img, img2, CV_BGR2HSV);
    split(img2, hsv);

    imshow("img", img);
    imshow("h", hsv[0]);
    imshow("s", hsv[1]);
    imshow("v", hsv[2]);

    for (;;)
        if ((char)waitKey(1) == 'q') break;

    return 0;
}

int example6() {

    namedWindow("img", WINDOW_KEEPRATIO);
    namedWindow("h", WINDOW_KEEPRATIO);
    namedWindow("s", WINDOW_KEEPRATIO);
    namedWindow("v", WINDOW_KEEPRATIO);

    Mat img = imread("img/rgbcube_kBKG.png", IMREAD_COLOR);
    Mat img2;
    std::vector<Mat> hsv;
    cvtColor(img, img2, CV_BGR2HSV);
    split(img2, hsv);

    imshow("img", img);
    imshow("h", hsv[0]);
    imshow("s", hsv[1]);
    imshow("v", hsv[2]);

    for (;;)
        if ((char)waitKey(1) == 'q') break;

    return 0;
}

int example7() {

    namedWindow("img", WINDOW_KEEPRATIO);

    int rows = 1e3;
    int radius = (int)(rows/4);
    int bx = (int)(rows/2), by = (int)(rows/2) - (int)(radius/2);
    int gx =(int)(rows/2) - radius/2;
    int gy =(int)(rows/2) + radius/2;
    int rx =(int)(rows/2) + radius/2;
    int ry =(int)(rows/2) + radius/2;
    Mat img;
    std::vector<Mat> bgr;
    bgr.push_back(createWhiteDisk(rows, rows, bx, by, radius));
    bgr.push_back(createWhiteDisk(rows, rows, gx, gy, radius));
    bgr.push_back(createWhiteDisk(rows, rows, rx, ry, radius));
    merge(bgr, img);
    img = scaleImage2_uchar(img);
    imshow("img", img);
    for (;;)
        if ((char)waitKey(1) == 'q') break;

    return 0;
}

int example8() {

    namedWindow("img", WINDOW_KEEPRATIO);

    int rows = 1e3;
    int radius = (int)(rows/4);
    int bx = (int)(rows/2), by = (int)(rows/2) - (int)(radius/2);
    int gx =(int)(rows/2) - radius/2;
    int gy =(int)(rows/2) + radius/2;
    int rx =(int)(rows/2) + radius/2;
    int ry =(int)(rows/2) + radius/2;
    Mat img;
    std::vector<Mat> bgr;
    bgr.push_back(createWhiteDisk(rows, rows, bx, by, radius));
    bgr.push_back(createWhiteDisk(rows, rows, gx, gy, radius));
    bgr.push_back(createWhiteDisk(rows, rows, rx, ry, radius));
    merge(bgr, img);
    img = scaleImage2_uchar(img);
    img = Scalar(255, 255, 255) - img;
    imshow("img", img);
    for (;;)
        if ((char)waitKey(1) == 'q') break;

    return 0;
}

int example9() {

    namedWindow("img", WINDOW_KEEPRATIO);
    namedWindow("img2", WINDOW_KEEPRATIO);

    Mat img = imread("img/baboon.png", IMREAD_COLOR);
    Mat img2;
    int wsize = 1;

    createTrackbar("wsize", "img2", &wsize, 50, 0, 0);

    for (;;) {
        blur(img, img2, Size(wsize + 1, wsize + 1), Point(-1, -1), BORDER_DEFAULT);
        imshow("img", img);
        imshow("img2", img2);
        if ((char)waitKey(1) == 'q') break;
    }

    return 0;
}

int example10() {

    namedWindow("img", WINDOW_KEEPRATIO);
    namedWindow("img2", WINDOW_KEEPRATIO);

    Mat img = imread("img/baboon.png", IMREAD_COLOR);
    Mat img2;
    int wsize = 1;

    createTrackbar("wsize", "img2", &wsize, 10, 0, 0);

    for (;;) {
        Laplacian(img, img2, CV_16S, 2*wsize+1, 1, 0, BORDER_DEFAULT);
        imshow("img", img);
        imshow("img2", img2);
        if ((char)waitKey(1) == 'q') break;
    }

    return 0;
}

int example11() {

    namedWindow("img", WINDOW_KEEPRATIO);
    namedWindow("img2", WINDOW_KEEPRATIO);

    Mat img = imread("img/baboon.png", IMREAD_COLOR);
    Mat img2;

    int sp = 10;
    int sr = 100;
    int maxLevel = 1;
    TermCriteria criteria = TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 5, 1);

    createTrackbar("maxLevel", "img2", &maxLevel, 5, 0, 0);
    createTrackbar("sr", "img2", &sr, 200, 0, 0);
    createTrackbar("sp", "img2", &sp, 20, 0, 0);

    for (;;) {
        pyrMeanShiftFiltering(img, img2, sp, sr, maxLevel, criteria);
        imshow("img", img);
        imshow("img2", img2);
        if ((char)waitKey(1) == 'q') break;
    }

    return 0;
}

int example12() {
    namedWindow("img", WINDOW_KEEPRATIO);
    namedWindow("img2", WINDOW_KEEPRATIO);
    namedWindow("img3", WINDOW_KEEPRATIO);

    Mat img = imread("img/calculator.tif", IMREAD_GRAYSCALE);
    Mat img2, img3;
    int size = 0, size2 = 0;

    createTrackbar("size", "img2", &size, 30);
    createTrackbar("size2", "img3", &size2, 21);

    Mat element, element2;

    for (;;) {
        element = getStructuringElement(MORPH_RECT, Size( 2*size + 1, 2*size + 1));
        element2 = getStructuringElement(MORPH_RECT, Size( 2*size2 + 1, 2*size2+1 ));

        morphologyEx(img, img2, MORPH_OPEN, element);
        morphologyEx(img2, img3, MORPH_TOPHAT, element2);

        imshow("img", img);
        imshow("img2", img2);
        imshow("img3", img3);

        if ((char)waitKey(1) == 'q') break;
    }

    return 0;
}

int example13() {
    namedWindow("img", WINDOW_KEEPRATIO);

    Mat img = imread("img/wirebond.tif", IMREAD_GRAYSCALE);
    Mat img2;

    Laplacian(img, img2, CV_32F, 1, 1, 0);
    img2 = abs(img2);

    imshow("img", scaleImage2_uchar(img2));

    for (;;) {

        if ((char)waitKey(1) == 'q') break;
    }

    return 0;
}

int example14() {
    namedWindow("img", WINDOW_KEEPRATIO);

    Mat img = imread("img/building.tif", IMREAD_GRAYSCALE);
    Mat img2, gx, gy, abs_gx, abs_gy;

    Sobel(img, gx, CV_32F, 1, 0, 3);
    Sobel(img, gy, CV_32F, 0, 1, 3);
    convertScaleAbs(gx, abs_gx);
    convertScaleAbs(gy, abs_gy);
    img2 = abs_gx + abs_gy;

    for (;;) {
        imshow("img", scaleImage2_uchar(img));
        imshow("gx", scaleImage2_uchar(abs_gx));
        imshow("gy", scaleImage2_uchar(abs_gy));
        imshow("img2", scaleImage2_uchar(img2));
        if ((char)waitKey(1) == 'q') break;
    }

    return 0;
}

int main(int argc, char *argv[])
{
    return example14();
}
