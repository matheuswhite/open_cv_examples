#include "utils.h"

int useWebcam() {
    namedWindow("img", WINDOW_KEEPRATIO);

    Mat img;
    VideoCapture vid;
    vid.open(0);
    if (!vid.isOpened()) return 1;

    for(;;)
    {
        vid >> img;

        imshow("img", img);

        if (waitKey(5) == 'q') break;
    }

    return 0;
}

int splitImage() {
    namedWindow("img", WINDOW_KEEPRATIO);
    namedWindow("b", WINDOW_KEEPRATIO);
    namedWindow("g", WINDOW_KEEPRATIO);
    namedWindow("r", WINDOW_KEEPRATIO);

    Mat img = imread("img/lena.png", IMREAD_COLOR);
    vector<Mat> bgr;

    split(img, bgr);

    imshow("img", img);
    imshow("b", bgr[0]);
    imshow("g", bgr[1]);
    imshow("r", bgr[2]);

    waitKey(0);

    return 0;
}

int useThreshold() {
    namedWindow("bin", WINDOW_KEEPRATIO);
    namedWindow("gray", WINDOW_KEEPRATIO);


    Mat img = imread("img/lena.png", IMREAD_GRAYSCALE);
    Mat bin;
    int thresh = 127;

    createTrackbar("Threshold", "bin", &thresh, 256, 0, 0);
    for(;;)
    {
        threshold(img, bin, thresh, 256, CV_THRESH_BINARY);

        if (img.empty() == true) return -1;

        imshow("bin", bin);
        imshow("img", img);

        if ((char)waitKey(5) == 'q') return 0;
    }

    return 0;
}

int useGaussianNoise(Utils *utils) {
    namedWindow("noise", WINDOW_KEEPRATIO);
    namedWindow("noise-hist", WINDOW_KEEPRATIO);
    namedWindow("img", WINDOW_KEEPRATIO);
    namedWindow("img-noise", WINDOW_KEEPRATIO);

    Mat img = imread("img/lena.png", IMREAD_GRAYSCALE);
    Mat noise = Mat::zeros(300, 300, CV_8U);
    Mat noise_hist;

    randn(noise, 50, 3);
    noise_hist = utils->computeHistogram1C(noise);

    imshow("img", img);
    img += noise;

    imshow("img-noise", img);
    imshow("nosie", noise);
    imshow("noise-hist", noise_hist);

    waitKey(0);

    return 0;
}

int useWhiteNoise(Utils *utils) {
    namedWindow("noise", WINDOW_KEEPRATIO);
    namedWindow("noise-hist", WINDOW_KEEPRATIO);
    namedWindow("img", WINDOW_KEEPRATIO);
    namedWindow("img-noise", WINDOW_KEEPRATIO);

    Mat img = imread("img/lena.png", IMREAD_GRAYSCALE);
    Mat noise = Mat::zeros(300, 300, CV_8U);
    Mat noise_hist;

    randu(noise, 0, 255);
    noise_hist = utils->computeHistogram1C(noise);

    imshow("img", img);
    img += noise;

    imshow("img-noise", img);
    imshow("nosie", noise);
    imshow("noise-hist", noise_hist);

    waitKey(0);

    return 0;
}

int convertColorType() {
    namedWindow("img", WINDOW_KEEPRATIO);
    namedWindow("img2", WINDOW_KEEPRATIO);

    Mat img = imread("img/lena.png", IMREAD_COLOR);
    Mat img2;

    cvtColor(img, img2, CV_RGB2GRAY);

    imshow("img", img);
    imshow("img2", img2);

    waitKey(0);

    return 0;
}

int negativeImage() {
    namedWindow("img", WINDOW_KEEPRATIO);
    namedWindow("img2", WINDOW_KEEPRATIO);

    Mat img = imread("img/lena.png", IMREAD_GRAYSCALE);
    Mat img2;

    img2 = 255 - img;

    imshow("img", img);
    imshow("img2", img2);

    waitKey(0);

    return 0;
}

int channelSum() {
    namedWindow("img", WINDOW_KEEPRATIO);
    namedWindow("img2", WINDOW_KEEPRATIO);

    Mat img = imread("img/lena.png", IMREAD_COLOR);
    Mat img2;

    img2 = img + Scalar(150, 0, 0);

    imshow("img", img);
    imshow("img2", img2);

    waitKey(0);

    return 0;
}

int matrixCreation() {
    namedWindow("img", WINDOW_KEEPRATIO);

    Mat img = 127 + Mat::zeros(30, 30, CV_8U);

    imshow("img", img);

    waitKey(0);

    return 0;
}

int imagesDiff() {
    namedWindow("img", WINDOW_KEEPRATIO);
    namedWindow("img2", WINDOW_KEEPRATIO);
    namedWindow("diff", WINDOW_KEEPRATIO);
    namedWindow("absdiff", WINDOW_KEEPRATIO);

    Mat img = imread("img/lena.png", IMREAD_COLOR);
    Mat img2 = imread("img/baboon.png", IMREAD_COLOR);
    Mat diff, _absdiff;

    absdiff(img, img2, _absdiff);
    diff = img - img2;

    imshow("img", img);
    imshow("img2", img2);

    waitKey(0);

    return 0;
}

int bitwiseOps() {
    namedWindow("utk", WINDOW_KEEPRATIO);
    namedWindow("gt", WINDOW_KEEPRATIO);
    namedWindow("and", WINDOW_KEEPRATIO);
    namedWindow("or", WINDOW_KEEPRATIO);
    //namedWindow("xor", WINDOW_KEEPRATIO);
    namedWindow("not", WINDOW_KEEPRATIO);

    Mat utk = imread("img/utk.tif", IMREAD_COLOR);
    Mat gt = imread("img/gt.tif", IMREAD_COLOR);
    Mat _and, _or, _not;

    _and = utk & gt;
    _or = utk | gt;
    //_xor = utk ^ gt;
    _not = ~utk;

    imshow("utk", utk);
    imshow("gt", gt);
    imshow("and", _and);
    imshow("or", _or);
    //imshow("xor", _xor);
    imshow("not", _not);

    waitKey(0);

    return 0;
}

int cutImage() {
    namedWindow("img", WINDOW_KEEPRATIO);
    namedWindow("img2", WINDOW_KEEPRATIO);

    Mat img = imread("img/lena.png", IMREAD_COLOR);
    Mat img2(img, Range(0, 400), Range(0, 200));

    imshow("img", img);
    imshow("img2", img2);

    waitKey(0);

    return 0;
}

int gradient() {
    namedWindow("img", WINDOW_KEEPRATIO);
    namedWindow("img2", WINDOW_KEEPRATIO);

    Mat img = 127 * Mat::ones(256, 256, CV_8U);
    Mat img2 = img.clone();

    for (int r = 0; r < img.rows; ++r) {
        img.row(r).setTo(r);
    }

    for (int c = 0; c < img.cols; ++c) {
        img.col(c).setTo(c);
    }

    imshow("img", img);
    imshow("img2", img2);

    waitKey(0);

    return 0;
}

int powerTransform() {
    namedWindow("img", WINDOW_KEEPRATIO);
    namedWindow("img2", WINDOW_KEEPRATIO);

    int value = 2;
    float pot;
    createTrackbar("root", "img2", &value, 10, 0, 0);

    Mat img = imread("img/spectrum.tif", IMREAD_GRAYSCALE);

    imshow("img", img);

    for(;;) {

        Mat img2 = Mat::zeros(img.rows, img.cols, CV_64F);

        for (int x = 0; x < img2.cols; ++x) {
            for (int y = 0; y < img2.rows; ++y) {
                Scalar intensity = img.at<uchar>(y, x);
                if (value != 0)
                    pot = 1/(float)value;
                double intensity_new = pow((double) intensity.val[0], pot);
                img2.at<double>(y, x) = intensity_new;
            }
        }

        /// Normalization
        normalize(img2, img2, 1, 0, NORM_MINMAX);
        img2 = 255 * img2;
        img2.convertTo(img2, CV_8U);

        imshow("img2", img2);

        if (waitKey(10) == 'q') break;
    }

    return 0;
}

int logTransform() {
    namedWindow("img", WINDOW_KEEPRATIO);
    namedWindow("img2", WINDOW_KEEPRATIO);

    Mat img = imread("img/spectrum.tif", IMREAD_GRAYSCALE);
    Mat img2 = Mat::zeros(img.rows, img.cols, CV_64F);

    for (int x = 0; x < img2.cols; ++x) {
        for (int y = 0; y < img2.rows; ++y) {
            Scalar intensity = img.at<uchar>(y, x);
            double intensity_new = 50 * log(1 + intensity.val[0]);
            img2.at<double>(y, x) = intensity_new;
        }
    }

    normalize(img2, img2, 1, 0, NORM_MINMAX);
    img2 = 255 * img2;
    img2.convertTo(img2, CV_8U);

    imshow("img", img);
    imshow("img2", img2);

    for(;;) if (waitKey(10) == 'q') break;

    return 0;
}

int linearRangeTransform() {
    namedWindow("img", WINDOW_KEEPRATIO);

    Mat img = imread("imgkidney.tif", IMREAD_GRAYSCALE);
    img = 150 < img & img < 240;
    imshow("img", img);

    for(;;)
        if (waitKey(10) == 'q') break;

    return 0;
}

int bitLayerSplit() {
    namedWindow("img", WINDOW_KEEPRATIO);
    namedWindow("img2", WINDOW_KEEPRATIO);
    namedWindow("img3", WINDOW_KEEPRATIO);

    Mat img = imread("img/dollar.tif", IMREAD_GRAYSCALE);
    Mat img2, img3;
    int slice = 7;
    createTrackbar("slice", "img2", &slice, 7, 0, 0);

    for(;;) {
        bitwise_and(img, (uchar)(pow(2, slice+1) - 1), img2);
        bitwise_and(img, (uchar)0x80, img3);
        imshow("img", img);
        imshow("img2", img2);
        imshow("img3", img3);
        if ((char)waitKey(1) == 'q') break;
    }

    return 0;
}

int tresholdTypes() {
    namedWindow("img", WINDOW_KEEPRATIO);
    namedWindow("img2", WINDOW_KEEPRATIO);

    Mat img = imread("img/kidney.tif", IMREAD_GRAYSCALE);
    Mat img2;
    int threshType = 0;
    int thresh = 127;
    createTrackbar("thresholdType", "img2", &threshType, 4, 0, 0);
    createTrackbar("thresh", "img2", &thresh, 255, 0, 0);

    for(;;) {
        threshold(img, img2, thresh, 255, threshType);
        imshow("img", img);
        imshow("img2", img2);

        if ((char)waitKey(1) == 'q') break;
    }

    return 0;
}

int equalizeHistogram(Utils *utils) {
    namedWindow("img", WINDOW_KEEPRATIO);
    namedWindow("img2", WINDOW_KEEPRATIO);
    namedWindow("hist", WINDOW_KEEPRATIO);
    namedWindow("hist2", WINDOW_KEEPRATIO);

    Mat img = imread("img/pollen_washedout.tif", IMREAD_GRAYSCALE);
    Mat img2;
    Mat hist;
    Mat hist2;
    equalizeHist(img ,img2);
    hist = utils->computeHistogram1C(img);
    hist2 = utils->computeHistogram1C(img2);
    imshow("img", img);
    imshow("img2", img2);
    imshow("hist", hist);
    imshow("hist2", hist2);

    for(;;) {
        if ((char)waitKey(1) == 'q') break;
    }

    return 0;
}



int piecewiseLinearTransform(Utils *utils) {

    namedWindow("img", WINDOW_KEEPRATIO);
    namedWindow("img2", WINDOW_KEEPRATIO);
    namedWindow("hist", WINDOW_KEEPRATIO);
    namedWindow("hist2", WINDOW_KEEPRATIO);
    namedWindow("control", WINDOW_KEEPRATIO);

    int x1 = 65, y1 = 65, x2 = 195, y2 = 195;

    createTrackbar("x1", "control", &x1, 255, 0, 0);
    createTrackbar("y1", "control", &y1, 255, 0, 0);
    createTrackbar("x2", "control", &x2, 255, 0, 0);
    createTrackbar("y2", "control", &y2, 255, 0, 0);

    Mat img = imread("img/pollen.jpg", IMREAD_GRAYSCALE);
    Mat hist, hist2;

    for(;;)
    {
        /// Piecewise linear core
        Mat img2(img.rows, img.cols, CV_8U);

        for (int i = 0; i < img2.cols; ++i) {
            for (int j = 0; j < img2.rows; ++j) {
                Scalar intensity = img.at<uchar>(j, i);
                if (0 <= intensity.val[0] && intensity.val[0] < x1) {
                    if (x1 == 0) continue;

                    double m = (double)y1/x1;
                    img2.at<uchar>(j, i) = m * intensity.val[0];
                }
                else if (x1 <= intensity.val[0] && intensity.val[0] < x2) {
                    if (x2-x1 == 0) continue;

                    double m = ((double)(y2-y1))/(x2-x1);
                    double b = - m*x1 + y1;
                    img2.at<uchar>(j, i) = m * intensity.val[0] + b;
                }
                else if (x2 <= intensity.val[0]) {
                    if (x2 == 255) continue;

                    double m = ((double)(255-y2))/(255-x2);
                    double b = - m*x2 + y2;
                    img2.at<uchar>(j, i) = m * intensity.val[0] + b;
                }
            }
        }

        /// Draw control image
        Mat control(256, 256, CV_8U, Scalar(255));

        line( control, Point( 0, 255), Point( x1, 255 - y1), Scalar( 0 ), 2, 8, 0  );
        line( control, Point( x1, 255 - y1), Point( x2, 255 - y2), Scalar( 0 ), 2, 8, 0  );
        line( control, Point( x2, 255 - y2), Point( 255, 0), Scalar( 0 ), 2, 8, 0  );

        circle( control, Point( x1, 255 - y1 ), 4, Scalar( 0 ), 2, 8, 0 );
        circle( control, Point( x2, 255 - y2 ), 4, Scalar( 0 ), 2, 8, 0 );

        hist = utils->computeHistogram1C(img);
        hist2 = utils->computeHistogram1C(img2);

        imshow("img", img);
        imshow("img2", img2);
        imshow("hist", hist);
        imshow("hist2", hist2);
        imshow("control", control);

        if (waitKey(5) == 'q') break;
    }

    return 0;
}

int equalizeHistogramRange() {
    namedWindow("img", WINDOW_KEEPRATIO);
    namedWindow("img2", WINDOW_KEEPRATIO);

    Mat img = imread("img/squares_noisy.tif", IMREAD_GRAYSCALE);
    Mat img2 = Mat::zeros(img.rows, img.cols, img.type());
    int wsize = 1;

    for (int x = wsize; x < img.cols - wsize; ++x) {
        for (int y = wsize; y < img.rows - wsize; ++y) {
            equalizeHist(img(Range(y-wsize, y+wsize), Range(x-wsize, x+wsize)),
                         img2(Range(y-wsize, y+wsize), Range(x-wsize, x+wsize)));
        }
    }

    imshow("img", img);
    imshow("img2", img2);

    for(;;)
        if ((char)waitKey(5) == 'q') break;

    return 0;
}

int blurVerticalHorizontal() {
    namedWindow("img", WINDOW_KEEPRATIO);
    namedWindow("img2", WINDOW_KEEPRATIO);

    Mat img = imread("img/lena.png", IMREAD_COLOR);
    Mat img2;
    int ksizex = 3, ksizey = 3;

    createTrackbar("ksizex", "img2", &ksizex, 63, 0, 0);
    createTrackbar("ksizey", "img2", &ksizey, 63, 0, 0);

    for(;;) {
        if (ksizex < 1) ksizex = 1;
        if (ksizey < 1) ksizey = 1;
        blur(img, img2, Size(ksizex, ksizey), Point(-1, -1), BORDER_DEFAULT);
        imshow("img", img);
        imshow("img2", img2);
        if ((char)waitKey(5) == 'q') break;
    }

    return 0;
}

int medianBlurFilter() {
    namedWindow("img", WINDOW_KEEPRATIO);
    namedWindow("img2", WINDOW_KEEPRATIO);
    namedWindow("img3", WINDOW_KEEPRATIO);

    Mat img = imread("img/lena.png", IMREAD_COLOR);
    Mat noise = Mat::zeros(img.rows, img.cols, CV_8U);
    Mat img2, img3, salt, pepper;
    int ksize = 0;
    int amount = 5;

    createTrackbar("ksize", "img3", &ksize, 15, 0, 0);
    createTrackbar("amount", "img2", &amount, 120, 0, 0);
    randu(noise, 0, 255);

    for(;;) {
        img2 = img.clone();
        salt = noise > 255 - amount;
        pepper = noise < amount;
        img2.setTo(255, salt);
        img2.setTo(0, pepper);

        medianBlur(img2, img3, (ksize+1)*2 - 1);
        imshow("img", img);
        imshow("img2", img2);
        imshow("img3", img3);
        if ((char)waitKey(5) == 'q') break;
    }

    return 0;
}



int sobelFilter(Utils *utils) {
    namedWindow("img", WINDOW_KEEPRATIO);
    namedWindow("gx", WINDOW_KEEPRATIO);
    namedWindow("gy", WINDOW_KEEPRATIO);
    namedWindow("g", WINDOW_KEEPRATIO);

    Mat img = imread("img/lena.png", IMREAD_GRAYSCALE);
    Mat gx, gy, g;
    img.convertTo(img, CV_32F, 1, 0);
    normalize(img, img, 1, 0, NORM_MINMAX);
    Mat kx = (Mat_<float>(3,3) <<
              -1, 0, 1,
              -2, 0, 2,
              -1, 0, 1
              );
    Mat ky = (Mat_<float>(3,3) <<
              -1, -2, -1,
              0, 0, 0,
              1, 2, 1
              );
    filter2D(img, gx, CV_32F, kx, Point(-1, -1), 0, BORDER_DEFAULT);
    filter2D(img, gy, CV_32F, ky, Point(-1, -1), 0, BORDER_DEFAULT);
    g = abs(gx) + abs(gy);
    gx = utils->scaleImage2_uchar(gx);
    gy = utils->scaleImage2_uchar(gy);
    g = utils->scaleImage2_uchar(g);
    for(;;) {
        imshow("img", img);
        imshow("gx", gx);
        imshow("gy", gy);
        imshow("g", g);
        if ((char)waitKey(5) == 'q') break;
    }

    return 0;
}

int fourBlurTypes() {
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

int easySobelFilter(Utils *utils) {
    namedWindow("img", WINDOW_KEEPRATIO);
    namedWindow("gx", WINDOW_KEEPRATIO);
    namedWindow("gy", WINDOW_KEEPRATIO);
    namedWindow("g", WINDOW_KEEPRATIO);

    Mat img = imread("img/lena.png", IMREAD_GRAYSCALE);
    Mat gx, gy, g;
    spatialGradient(img, gx, gy, 3, BORDER_DEFAULT);
    g = abs(gx) + abs(gy);
    gx = utils->scaleImage2_uchar(gx);
    gy = utils->scaleImage2_uchar(gy);
    g  = utils->scaleImage2_uchar(g);

    for (;;) {
        imshow("img", img);
        imshow("gx", gx);
        imshow("gy", gy);
        imshow("g", g);

        if ((char)waitKey(1)=='q') break;
    }

    return 0;
}



int easySobelFilter2(Utils *utils) {
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
        imshow("img", utils->scaleImage2_uchar(img));
        imshow("gx", utils->scaleImage2_uchar(gx));
        imshow("gy", utils->scaleImage2_uchar(gy));
        imshow("g", utils->scaleImage2_uchar(g));

        if ((char)waitKey(1)=='q') break;
    }

    return 0;
}

int laplacianFilter(Utils *utils) {
    namedWindow("img", WINDOW_KEEPRATIO);
    namedWindow("img2", WINDOW_KEEPRATIO);
    namedWindow("img3", WINDOW_KEEPRATIO);
    namedWindow("hist", WINDOW_KEEPRATIO);


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
        Mat hist = utils->computeHistogram1C(img3);
        add(img, -(factor/100.0)*img2, img3, noArray(), CV_8U);
        imshow("img", utils->scaleImage2_uchar(img));
        imshow("img2", utils->scaleImage2_uchar(img2));
        imshow("img3", utils->scaleImage2_uchar(img3));
        imshow("hist", utils->scaleImage2_uchar(hist));
        if ((char)waitKey(5) == 'q') break;
    }

    return 0;
}

int easyLaplacianFilter(Utils *utils) {
    namedWindow("img", WINDOW_KEEPRATIO);
    namedWindow("img2", WINDOW_KEEPRATIO);
    namedWindow("lap", WINDOW_KEEPRATIO);


    Mat img = imread("img/lena.png", IMREAD_GRAYSCALE);
    Mat lap, img2;
    Laplacian(img, lap, CV_32F, 1, 1, 0);
    add(img, -lap, img2, noArray(), CV_8U);

    for(;;) {
        imshow("img", img);
        imshow("img2", img2);
        imshow("lap", utils->scaleImage2_uchar(lap));
        if ((char)waitKey(5) == 'q') break;
    }

    return 0;
}



int dftMethod(Utils *utils) {
    namedWindow("img", WINDOW_KEEPRATIO);
    namedWindow("planes_0", WINDOW_KEEPRATIO);
    namedWindow("planes_1", WINDOW_KEEPRATIO);


    Mat img = imread("img/lena.png", IMREAD_GRAYSCALE);
    Mat planes[] = {Mat_<float>(img), Mat::zeros(img.size(), CV_32F)};
    Mat img2;

    merge(planes, 2, img2);
    dft(img2, img2);
    split(img2, planes);
    normalize(planes[0], planes[0], 1, 0, NORM_MINMAX);
    normalize(planes[1], planes[1], 1, 0, NORM_MINMAX);

    for(;;) {
        imshow("img", utils->scaleImage2_uchar(img));
        imshow("planes_0", utils->fftshift(planes[0]));
        imshow("planes_1", utils->fftshift(planes[1]));
        if ((char)waitKey(5) == 'q') break;
    }

    return 0;
}

int dftMagnitude(Utils *utils) {
    namedWindow("img", WINDOW_KEEPRATIO);
    namedWindow("planes_0", WINDOW_KEEPRATIO);
    namedWindow("planes_1", WINDOW_KEEPRATIO);
    namedWindow("mag", WINDOW_KEEPRATIO);


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
        imshow("img", utils->scaleImage2_uchar(img));
        imshow("planes_0", utils->fftshift(planes[0]));
        imshow("planes_1", utils->fftshift(planes[1]));
        imshow("mag", utils->fftshift(utils->scaleImage2_uchar(mag)));
        if ((char)waitKey(5) == 'q') break;
    }

    return 0;
}

int whiteDisk(Utils *utils) {
    namedWindow("disk", WINDOW_KEEPRATIO);


    Mat disk = Mat::zeros(200, 200, CV_32F);

    int xc = 100;
    int yc = 100;
    int radius = 20;

    createTrackbar("xc", "disk", &xc, disk.cols, 0);
    createTrackbar("yc", "disk", &yc, disk.rows, 0);
    createTrackbar("radius", "disk", &radius, disk.cols, 0);

    for(;;) {
        disk = Mat::zeros(200, 200, CV_32F);

        for (int x = 0; x < disk.cols; ++x) {
            for (int y = 0; y < disk.rows; ++y) {
                if ((x-xc) * (x-xc) + (y-yc) * (y-yc) <= radius * radius)
                    disk.at<float>(x, y) = 1.0;
            }
        }

        imshow("disk", utils->scaleImage2_uchar(disk));
        if ((char)waitKey(5) == 'q') break;
    }

    return 0;
}



int easyWhiteDisk(Utils *utils) {
    namedWindow("disk", WINDOW_KEEPRATIO);


    Mat disk = Mat::zeros(200, 200, CV_32F);

    int xc = 100;
    int yc = 100;
    int radius = 20;

    createTrackbar("xc", "disk", &xc, disk.cols, 0);
    createTrackbar("yc", "disk", &yc, disk.rows, 0);
    createTrackbar("radius", "disk", &radius, disk.cols, 0);

    for(;;) {
        disk = utils->createWhiteDisk(200, 200, xc, yc, radius);

        imshow("disk", utils->scaleImage2_uchar(disk));
        if ((char)waitKey(5) == 'q') break;
    }

    return 0;
}

int lowPassFilterDFT(Utils *utils) {
    namedWindow("img", WINDOW_KEEPRATIO);
    namedWindow("img2", WINDOW_KEEPRATIO);
    namedWindow("planes_0", WINDOW_KEEPRATIO);
    namedWindow("planes_1", WINDOW_KEEPRATIO);
    namedWindow("mask", WINDOW_KEEPRATIO);


    Mat img = imread("img/lena.png", IMREAD_GRAYSCALE);
    Mat img2 = img.clone();

    int radius = 50;

    createTrackbar("radius", "mask", &radius, img2.cols, 0, 0);

    for(;;) {
        Mat mask = utils->createWhiteDisk(img2.rows, img2.cols, (int)img2.cols/2, (int)img2.rows/2, radius);
        mask = utils->fftshift(mask);

        Mat planes[] =  {Mat_<float>(img), Mat::zeros(img.size(), CV_32F)};

        merge(planes, 2, img2);
        dft(img2, img2);
        split(img2, planes);

        multiply(planes[0], mask, planes[0]);
        multiply(planes[1], mask, planes[1]);
        merge(planes, 2, img2);
        idft(img2, img2, DFT_REAL_OUTPUT);
        img2 = utils->fftshift(img2);

        imshow("img", utils->scaleImage2_uchar(img));
        imshow("planes_0", planes[0]);
        imshow("planes_1", planes[1]);
        imshow("img2", utils->fftshift(utils->scaleImage2_uchar(img2)));
        imshow("mask", mask);
        if ((char)waitKey(5) == 'q') break;
    }

    return 0;
}

int highPassFilterDFT(Utils *utils) {
    namedWindow("img", WINDOW_KEEPRATIO);
    namedWindow("img2", WINDOW_KEEPRATIO);
    namedWindow("planes_0", WINDOW_KEEPRATIO);
    namedWindow("planes_1", WINDOW_KEEPRATIO);
    namedWindow("mask", WINDOW_KEEPRATIO);


    Mat img = imread("img/lena.png", IMREAD_GRAYSCALE);
    Mat img2 = img.clone();

    int radius = 50;

    createTrackbar("radius", "mask", &radius, img2.cols, 0, 0);

    for(;;) {
        Mat mask = utils->createWhiteDisk(img2.rows, img2.cols, (int)img2.cols/2, (int)img2.rows/2, radius);
        mask = utils->fftshift(mask);
        mask = 1 - mask;

        Mat planes[] =  {Mat_<float>(img), Mat::zeros(img.size(), CV_32F)};

        merge(planes, 2, img2);
        dft(img2, img2);
        split(img2, planes);

        multiply(planes[0], mask, planes[0]);
        multiply(planes[1], mask, planes[1]);
        merge(planes, 2, img2);
        idft(img2, img2, DFT_REAL_OUTPUT);
        img2 = utils->fftshift(img2);

        imshow("img", utils->scaleImage2_uchar(img));
        imshow("planes_0", planes[0]);
        imshow("planes_1", planes[1]);
        imshow("img2", utils->fftshift(utils->scaleImage2_uchar(img2)));
        imshow("mask", mask);
        if ((char)waitKey(5) == 'q') break;
    }

    return 0;
}

int cosineImage(Utils *utils) {
    namedWindow("img", WINDOW_KEEPRATIO);


    Mat img;
    int rows = 500;
    int cols = 500;
    int freq = 1;
    int theta = 2;

    createTrackbar("Freq", "img", &freq, 100, 0, 0);
    createTrackbar("Theta", "img", &theta, 100, 0, 0);

    for(;;) {
        img = utils->createCosineImg(rows, cols, (float)freq/1e3, (float)(2*CV_PI*theta/100.0));
        imshow("img", utils->scaleImage2_uchar(img));
        if ((char)waitKey(5) == 'q') break;
    }

    return 0;
}

int cosineNoise(Utils *utils) {
    namedWindow("img", WINDOW_KEEPRATIO);
    namedWindow("mag", WINDOW_KEEPRATIO);


    Mat img = imread("img/lena.png", IMREAD_GRAYSCALE);
    img.convertTo(img, CV_32F);
    img = img / 255.0;
    Mat noise;

    int rows = img.rows;
    int cols = img.cols;

    int freq = 1;
    int theta = 2;
    int gain = 1;

    createTrackbar("Freq", "img", &freq, 500, 0, 0);
    createTrackbar("Theta", "img", &theta, 100, 0, 0);
    createTrackbar("Gain", "img", &gain, 100, 0, 0);

    for(;;) {
        noise = utils->createCosineImg(rows, cols, (float)freq/1e3, (float)(2*CV_PI*theta/100.0));

        noise = img + (float)(gain/100.0) * noise;

        Mat img3 = noise.clone();
        Mat img2, mag;
        Mat planes[] =  {Mat_<float>(img3), Mat::zeros(img3.size(), CV_32F)};
        merge(planes, 2, img2);
        dft(img2, img2);
        split(img2, planes);
        magnitude(planes[0], planes[1], mag);
        mag += 1;
        log(mag, mag);

        imshow("img", utils->scaleImage2_uchar(noise));
        imshow("mag", utils->fftshift(utils->scaleImage2_uchar(mag)));
        if ((char)waitKey(5) == 'q') break;
    }

    return 0;
}

int sliptBGRChannelsShowColormap(Utils *utils) {

    namedWindow("baboon", WINDOW_KEEPRATIO);
    namedWindow("b", WINDOW_KEEPRATIO);
    namedWindow("g", WINDOW_KEEPRATIO);
    namedWindow("r", WINDOW_KEEPRATIO);

    Mat baboon = imread("img/baboon.png", IMREAD_COLOR);
    std::vector<Mat> bgr;
    split(baboon, bgr);

    imshow("baboon", baboon);
    imshow("b", utils->cvtImg2Colormap(bgr[0], COLORMAP_JET));
    imshow("g", utils->cvtImg2Colormap(bgr[1], COLORMAP_JET));
    imshow("r", utils->cvtImg2Colormap(bgr[2], COLORMAP_JET));

    for (;;)
        if ((char)waitKey(1) == 'q') break;

    return 0;
}

int sliptBGRChannelsNegative() {

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

int sliptYCrCbChannels() {

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

int sliptHSVChannels() {

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

int sliptHSVChannelsChips() {

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

int sliptHSVChannelsCube() {

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

int RGBdisks(Utils *utils) {

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
    bgr.push_back(utils->createWhiteDisk(rows, rows, bx, by, radius));
    bgr.push_back(utils->createWhiteDisk(rows, rows, gx, gy, radius));
    bgr.push_back(utils->createWhiteDisk(rows, rows, rx, ry, radius));
    merge(bgr, img);
    img = utils->scaleImage2_uchar(img);
    imshow("img", img);
    for (;;)
        if ((char)waitKey(1) == 'q') break;

    return 0;
}

int RGBdisksNegative(Utils *utils) {

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
    bgr.push_back(utils->createWhiteDisk(rows, rows, bx, by, radius));
    bgr.push_back(utils->createWhiteDisk(rows, rows, gx, gy, radius));
    bgr.push_back(utils->createWhiteDisk(rows, rows, rx, ry, radius));
    merge(bgr, img);
    img = utils->scaleImage2_uchar(img);
    img = Scalar(255, 255, 255) - img;
    imshow("img", img);
    for (;;)
        if ((char)waitKey(1) == 'q') break;

    return 0;
}

int BlurColor() {

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

int LaplacianColor() {

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

int meanShiftFilter() {

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

int LaplacianConvertScaleAbs(Utils *utils) {
    namedWindow("img", WINDOW_KEEPRATIO);

    Mat img = imread("img/wirebond.tif", IMREAD_GRAYSCALE);
    Mat img2;

    Laplacian(img, img2, CV_32F, 1, 1, 0);
    convertScaleAbs(img2, img2);

    imshow("img", utils->scaleImage2_uchar(img2));

    for (;;) {

        if ((char)waitKey(1) == 'q') break;
    }

    return 0;
}

int SobelConvertScaleAbs(Utils *utils) {
    namedWindow("img", WINDOW_KEEPRATIO);

    Mat img = imread("img/building.tif", IMREAD_GRAYSCALE);
    Mat img2, gx, gy, abs_gx, abs_gy;

    Sobel(img, gx, CV_32F, 1, 0, 3);
    Sobel(img, gy, CV_32F, 0, 1, 3);
    convertScaleAbs(gx, abs_gx);
    convertScaleAbs(gy, abs_gy);
    img2 = abs_gx + abs_gy;

    for (;;) {
        imshow("img", utils->scaleImage2_uchar(img));
        imshow("gx", utils->scaleImage2_uchar(abs_gx));
        imshow("gy", utils->scaleImage2_uchar(abs_gy));
        imshow("img2", utils->scaleImage2_uchar(img2));
        if ((char)waitKey(1) == 'q') break;
    }

    return 0;
}

int removeBrightnessOnTheCalculator(Utils *utils)
{
    namedWindow("img", WINDOW_KEEPRATIO);
    namedWindow("img2", WINDOW_KEEPRATIO);
    namedWindow("img3", WINDOW_KEEPRATIO);
    namedWindow("final", WINDOW_KEEPRATIO);

    Mat img = imread("img/calculator.tif", IMREAD_GRAYSCALE);
    Mat img2, img3, img4;
    int times = 70, times2 = 55, times3 = 30;

    createTrackbar("times", "img2", &times, 100);
    createTrackbar("times", "img3", &times2, 100);
    createTrackbar("times", "final", &times3, 100);

    Mat element, element2, element3, element4;

    for (;;) {
        element = getStructuringElement(MORPH_RECT, Size( 71, 1));
        element2 = getStructuringElement(MORPH_RECT, Size( 11, 1));
        element3 = getStructuringElement(MORPH_RECT, Size(3, 3));
        element4 = getStructuringElement(MORPH_RECT, Size(21, 1));

        //removing brightness
        img2 = utils->OpeningByReconstruction(img, element, element3, times);
        img3 = img - img2;
        //reconstructing the vertical structs
        img4 = utils->OpeningByReconstruction(img3, element2, element3, times2);
        morphologyEx(img4, img4, MORPH_DILATE, element4);
        img4 = min(img3, img4);
        //final reconstruction
        for (int i = 0; i < times3; ++i) {
            morphologyEx(img4, img4, MORPH_DILATE, element3);
            img4 = min(img4, img3);
        }

        imshow("img", img);
        imshow("img2", img2);
        imshow("img3", img3);
        imshow("final", img4);

        if ((char)waitKey(1) == 'q') break;
    }

    return 0;
}

void HoughTransformation() {
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

int equalizeHistRGB(Utils *utils) {

    namedWindow("img", WINDOW_KEEPRATIO);
    namedWindow("img2", WINDOW_KEEPRATIO);

    Mat img = imread("img/lena.png", IMREAD_COLOR);
    Mat img2;
    Mat hsv[3];
    int offset = 0;

    createTrackbar("offset", "img2", &offset, 127, 0);

    for(;;) {
        cvtColor(img, img2, CV_BGR2HSV);

        split(img2, hsv);
        equalizeHist(hsv[2], hsv[2]);
        hsv[2] += offset;
        merge(hsv, 3, img2);

        cvtColor(img2, img2, CV_HSV2BGR);

        imshow("img", img);
        imshow("img2", img2);

        if ((char)waitKey(5) == 'q') break;
    }

    return 0;
}

/*
 * (+) Dilatation
 * (-) Erosion
 * o Opening
 * * CLosing
 * # Hit-or-miss
 *
 * boundary             = A - (A (-) B)
 * hole filling         Xk = (Xk-1 (+) B) & ~A
 * connected components Xk = (Xk-1 (+) B) & ~A
 * convex hull Xi,k = (Xi,k-1 # Bi) | A
 *             Xi,0 = A
 * Thinning   = A & ~(A # B)
 * Thickening = A |  (A # B)
 * Skeletons
 *  for (0, K)
 *      s |= (A (-) kB) - (A (-) kB) o B
 *
