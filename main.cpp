#include "examples.h"

int example2() {
    Examples *e = new Examples();

    namedWindow("img", WINDOW_KEEPRATIO);
    namedWindow("gx", WINDOW_KEEPRATIO);
    namedWindow("gy", WINDOW_KEEPRATIO);
    namedWindow("g", WINDOW_KEEPRATIO);

    Mat img = imread("img/lena.png", IMREAD_GRAYSCALE);
    Mat gx, gy, g;
    Sobel(img, gx, CV_32F, 1, 0, 3);
    Sobel(img, gy, CV_32F, 0, 1, 3);
    g = abs(gx) + abs(gy);

    for (;;) {
        imshow("img", e->scaleImage2_uchar(img));
        imshow("gx", e->scaleImage2_uchar(gx));
        imshow("gy", e->scaleImage2_uchar(gy));
        imshow("g", e->scaleImage2_uchar(g));

        if ((char)waitKey(1)=='q') break;
    }

    return 0;
}

int example3() {
    namedWindow("img", WINDOW_KEEPRATIO);
    namedWindow("img2", WINDOW_KEEPRATIO);
    namedWindow("img3", WINDOW_KEEPRATIO);
    namedWindow("hist", WINDOW_KEEPRATIO);

    Examples *e = new Examples();
    Mat img = imread("img/lena.png", IMREAD_GRAYSCALE);
    Mat img2, img3;
    int factor = 5;
    img.convertTo(img, CV_32F);
    Mat kernel = (Mat_<float>(3,3) <<
              1.0,  1.0, 1.0,
              1.0, -8.0, 1.0,
              1.0,  1.0, 1.0);
    filter2D(img, img2, CV_32F, kernel, Point(-1, -1), 0, BORDER_DEFAULT);
    createTrackbar("factor", "img3", &factor, 100, 0, 0);

    for(;;) {
        Mat hist = e->computeHistogram1C(img3);
        add(img, -(factor/100.0)*img2, img3, noArray(), CV_8U);
        imshow("img", e->scaleImage2_uchar(img));
        imshow("img2", e->scaleImage2_uchar(img2));
        imshow("img3", e->scaleImage2_uchar(img3));
        imshow("hist", e->scaleImage2_uchar(hist));
        if ((char)waitKey(5) == 'q') break;
    }

    return 0;
}

int example4() {
    namedWindow("img", WINDOW_KEEPRATIO);
    namedWindow("img2", WINDOW_KEEPRATIO);
    namedWindow("lap", WINDOW_KEEPRATIO);

    Examples *e = new Examples();
    Mat img = imread("img/lena.png", IMREAD_GRAYSCALE);
    Mat lap, img2;
    Laplacian(img, lap, CV_32F, 1, 1, 0);
    add(img, -lap, img2, noArray(), CV_8U);

    for(;;) {
        imshow("img", img);
        imshow("img2", img2);
        imshow("lap", e->scaleImage2_uchar(lap));
        if ((char)waitKey(5) == 'q') break;
    }

    return 0;
}

Mat fftshift(const Mat &src) {
    Mat tmp = src.clone();
    Mat tmp2;

    tmp = tmp(Rect(0, 0, tmp.cols & -2, tmp.rows & -2));

    int cx = tmp.cols/2;
    int cy = tmp.rows/2;

    Mat q0(tmp, Rect(0, 0, cx, cy));
    Mat q1(tmp, Rect(cx, 0, cx, cy));
    Mat q2(tmp, Rect(0, cy, cx, cy));
    Mat q3(tmp, Rect(cx, cy, cx, cy));

    q1.copyTo(tmp2);
    q2.copyTo(q1);
    tmp2.copyTo(q2);

    q0.copyTo(tmp2);
    q3.copyTo(q0);
    tmp2.copyTo(q3);

    return tmp;
}

int example5() {
    namedWindow("img", WINDOW_KEEPRATIO);
    namedWindow("planes_0", WINDOW_KEEPRATIO);
    namedWindow("planes_1", WINDOW_KEEPRATIO);

    Examples *e = new Examples();
    Mat img = imread("img/lena.png", IMREAD_GRAYSCALE);
    Mat planes[] = {Mat_<float>(img), Mat::zeros(img.size(), CV_32F)};
    Mat img2;

    merge(planes, 2, img2);
    dft(img2, img2);
    split(img2, planes);
    normalize(planes[0], planes[0], 1, 0, NORM_MINMAX);
    normalize(planes[1], planes[1], 1, 0, NORM_MINMAX);

    for(;;) {
        imshow("img", e->scaleImage2_uchar(img));
        imshow("planes_0", fftshift(planes[0]));
        imshow("planes_1", fftshift(planes[1]));
        if ((char)waitKey(5) == 'q') break;
    }

    return 0;
}

int example6() {
    namedWindow("img", WINDOW_KEEPRATIO);
    namedWindow("planes_0", WINDOW_KEEPRATIO);
    namedWindow("planes_1", WINDOW_KEEPRATIO);
    namedWindow("mag", WINDOW_KEEPRATIO);

    Examples *e = new Examples();
    Mat img = imread("img/lena.png", IMREAD_GRAYSCALE);
    Mat planes[] = {Mat_<float>(img), Mat::zeros(img.size(), CV_32F)};
    Mat img2;

    merge(planes, 2, img2);
    dft(img2, img2);
    split(img2, planes);

    Mat mag;
    magnitude(planes[0], planes[1], mag);
    mag += 1;
    log(mag, mag);

    for(;;) {
        imshow("img", e->scaleImage2_uchar(img));
        imshow("planes_0", fftshift(planes[0]));
        imshow("planes_1", fftshift(planes[1]));
        imshow("mag", fftshift(e->scaleImage2_uchar(mag)));
        if ((char)waitKey(5) == 'q') break;
    }

    return 0;
}

int main(int argc, char *argv[])
{
    return example6();
}
