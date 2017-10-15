#include "utils.h"

int horseThinning(Utils *utils) {

    namedWindow("img", WINDOW_KEEPRATIO);
    namedWindow("img2", WINDOW_KEEPRATIO);

    Mat img = imread("img/horse.png", IMREAD_GRAYSCALE);
    Mat img2, hitormiss;
    Mat kernel[8] = {(Mat_<char>(3, 3) <<
                     -1, -1, -1,
                     0, 1, 0,
                     1, 1, 1),
                     (Mat_<char>(3, 3) <<
                     0, -1, -1,
                     1, 1, -1,
                     1, 1, 0),
                     (Mat_<char>(3, 3) <<
                     1, 0, -1,
                     1, 1, -1,
                     1, 0, -1),
                     (Mat_<char>(3, 3) <<
                     1, 1, 0,
                     1, 1, -1,
                     0, -1, -1),
                     (Mat_<char>(3, 3) <<
                     1, 1, 1,
                     0, 1, 0,
                     -1, -1, -1),
                     (Mat_<char>(3, 3) <<
                     0, 1, 1,
                     -1, 1, 1,
                     -1, -1, 0),
                     (Mat_<char>(3, 3) <<
                     -1, 0, 1,
                     -1, 1, 1,
                     -1, 0, 1),
                     (Mat_<char>(3, 3) <<
                     -1, -1, 0,
                     -1, 1, 1,
                     0,  1, 1),
                    };

    threshold(img, img, 127, 255, THRESH_BINARY);

    img2 = img.clone();
    img2 = ~img2;

    for (int i = 0; i < 8*50; ++i) {
        morphologyEx(img2, hitormiss, MORPH_HITMISS, kernel[(i%8)]);
        hitormiss = ~hitormiss;
        img2 = img2 & hitormiss;
    }

    imshow("img", ~img);
    imshow("img2", img2);

    for(;;)
        if ((char)waitKey(1) == 'q') break;

    return 0;
}

int sudoku(Utils *utils) {

    namedWindow("img", WINDOW_KEEPRATIO);
    namedWindow("img2", WINDOW_KEEPRATIO);

    int size = 1, thresh = 1;
    Mat img = imread("img/sudoku.png", IMREAD_GRAYSCALE);
    Mat img2, kernel;

    createTrackbar("size", "img2", &size, 100, 0);
    createTrackbar("thresh", "img2", &thresh, 254, 0);

    for(;;) {
        kernel = getStructuringElement(MORPH_RECT, Size((size*2)+1, (size*2)+1));
        morphologyEx(~img, img2, MORPH_TOPHAT, kernel);
        threshold(img2, img2, thresh, 255, THRESH_BINARY);
        img2 = ~img2;

        imshow("img", img);
        imshow("img2", img2);

        if ((char)waitKey(1) == 'q') break;
    }

    return 0;
}

int main(int argc, char *argv[])
{
    Utils *utils = new Utils();

    return sudoku(utils);
}
