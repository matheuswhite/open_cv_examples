#include "utils.h"

void hough() {
    namedWindow("img", WINDOW_KEEPRATIO);
    namedWindow("img2", WINDOW_KEEPRATIO);

    Mat img = imread("img/building.tif", IMREAD_GRAYSCALE);
    Mat img2, img3;

    Canny(img, img2, 50, 200, 3);
    cvtColor(img2, img3, CV_GRAY2BGR);

    vector<Vec4i> lines;
    HoughLinesP(img2, lines, 1, CV_PI/180, 50, 50, 10);
    for( size_t i = 0; i < lines.size(); i++ )
    {
        Vec4i l = lines[i];
        line( img3, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
    }

    imshow("img", img);
    imshow("img2", img3);

    for(;;)
        if ((char)waitKey(5) == 'q') break;

    return;
}

int main(int argc, char *argv[])
{
    Utils *utils = new Utils();

    hough();

    return 0;
}
