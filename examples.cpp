#include "examples.h"

Examples::Examples()
{

}

int Examples::useWebcam() {
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

int Examples::splitImage() {
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

int Examples::useThreshold() {
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

int Examples::useGaussianNoise() {
    namedWindow("noise", WINDOW_KEEPRATIO);
    namedWindow("noise-hist", WINDOW_KEEPRATIO);
    namedWindow("img", WINDOW_KEEPRATIO);
    namedWindow("img-noise", WINDOW_KEEPRATIO);

    Mat img = imread("img/lena.png", IMREAD_GRAYSCALE);
    Mat noise = Mat::zeros(300, 300, CV_8U);
    Mat noise_hist;

    randn(noise, 50, 3);
    noise_hist = this->computeHistogram1C(noise);

    imshow("img", img);
    img += noise;

    imshow("img-noise", img);
    imshow("nosie", noise);
    imshow("noise-hist", noise_hist);

    waitKey(0);

    return 0;
}

int Examples::useWhiteNoise() {
    namedWindow("noise", WINDOW_KEEPRATIO);
    namedWindow("noise-hist", WINDOW_KEEPRATIO);
    namedWindow("img", WINDOW_KEEPRATIO);
    namedWindow("img-noise", WINDOW_KEEPRATIO);

    Mat img = imread("img/lena.png", IMREAD_GRAYSCALE);
    Mat noise = Mat::zeros(300, 300, CV_8U);
    Mat noise_hist;

    randu(noise, 0, 255);
    noise_hist = this->computeHistogram1C(noise);

    imshow("img", img);
    img += noise;

    imshow("img-noise", img);
    imshow("nosie", noise);
    imshow("noise-hist", noise_hist);

    waitKey(0);

    return 0;
}

int Examples::convertColorType() {
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

int Examples::negativeImage() {
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

int Examples::channelSum() {
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

int Examples::matrixCreation() {
    namedWindow("img", WINDOW_KEEPRATIO);

    Mat img = 127 + Mat::zeros(30, 30, CV_8U);

    imshow("img", img);

    waitKey(0);

    return 0;
}

int Examples::imagesDiff() {
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

int Examples::bitwiseOps() {
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

int Examples::cutImage() {
    namedWindow("img", WINDOW_KEEPRATIO);
    namedWindow("img2", WINDOW_KEEPRATIO);

    Mat img = imread("img/lena.png", IMREAD_COLOR);
    Mat img2(img, Range(0, 400), Range(0, 200));

    imshow("img", img);
    imshow("img2", img2);

    waitKey(0);

    return 0;
}

int Examples::gradient() {
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

int Examples::powerTransform() {
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

int Examples::logTransform() {
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

int Examples::linearRangeTransform() {
    namedWindow("img", WINDOW_KEEPRATIO);

    Mat img = imread("imgkidney.tif", IMREAD_GRAYSCALE);
    img = 150 < img & img < 240;
    imshow("img", img);

    for(;;)
        if (waitKey(10) == 'q') break;

    return 0;
}

int Examples::bitLayerSplit() {
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

int Examples::tresholdTypes() {
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

int Examples::equalizeHistogram() {
    namedWindow("img", WINDOW_KEEPRATIO);
    namedWindow("img2", WINDOW_KEEPRATIO);
    namedWindow("hist", WINDOW_KEEPRATIO);
    namedWindow("hist2", WINDOW_KEEPRATIO);

    Mat img = imread("img/pollen_washedout.tif", IMREAD_GRAYSCALE);
    Mat img2;
    Mat hist;
    Mat hist2;
    equalizeHist(img ,img2);
    hist = computeHistogram1C(img);
    hist2 = computeHistogram1C(img2);
    imshow("img", img);
    imshow("img2", img2);
    imshow("hist", hist);
    imshow("hist2", hist2);

    for(;;) {
        if ((char)waitKey(1) == 'q') break;
    }

    return 0;
}

Mat Examples::computeHistogram1C (const Mat &src) {
    /// Establish the number of bins
    int histSize = 256;

    /// Set the ranges ( for B,G,R) )
    float range[] = { 0, 256 } ;
    const float* histRange = { range };

    bool uniform = true; bool accumulate = false;

    Mat b_hist/*, g_hist, r_hist*/;

    /// Compute the histograms:
    calcHist( &src, 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );


    // Draw the histograms for B, G and R
    int hist_w = 512; int hist_h = 400;
    int bin_w = cvRound( (double) hist_w/histSize );

    Mat histImage( hist_h, hist_w, CV_8UC1, Scalar( 0 ) );

    /// Normalize the result to [ 0, histImage.rows ]
    normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

    /// Draw for each channel
    for( int i = 1; i < histSize; i++ )
    {
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
              Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
              Scalar( 255 ), 2, 8, 0  );
    }

    return histImage;
}

Mat Examples::computeHistogram3C (const Mat &src) {
    /// Establish the number of bins
    int histSize = 256;

    /// Set the ranges ( for B,G,R) )
    float range[] = { 0, 256 } ;
    const float* histRange = { range };

    bool uniform = true; bool accumulate = false;

    Mat b_hist, g_hist, r_hist;
    Mat bgr[3];
    split(src, bgr);

    /// Compute the histograms:
    calcHist( &bgr[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &bgr[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &bgr[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );

    // Draw the histograms for B, G and R
    int hist_w = 512; int hist_h = 400;
    int bin_w = cvRound( (double) hist_w/histSize );

    Mat histImage( hist_h, hist_w, CV_32FC3, Scalar( 255, 255, 255 ) );

    /// Normalize the result to [ 0, histImage.rows ]
    normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

    /// Draw for each channel
    for( int i = 1; i < histSize; i++ )
    {
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
              Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
              Scalar( 255, 0, 0 ), 2, 8, 0  );

        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
              Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
              Scalar( 0, 255, 0 ), 2, 8, 0  );

        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
              Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
              Scalar( 0, 0, 255 ), 2, 8, 0  );
    }

    return histImage;
}

int Examples::piecewiseLinearTransform() {

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

        hist = computeHistogram1C(img);
        hist2 = computeHistogram1C(img2);

        imshow("img", img);
        imshow("img2", img2);
        imshow("hist", hist);
        imshow("hist2", hist2);
        imshow("control", control);

        if (waitKey(5) == 'q') break;
    }

    return 0;
}

int Examples::equalizeHistogramRange() {
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

int Examples::blurVerticalHorizontal() {
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

int Examples::medianBlurFilter() {
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

Mat Examples::scaleImage2_uchar(Mat &src) {
    Mat tmp = src.clone();
    if (src.type() != CV_32F)
        tmp.convertTo(tmp, CV_32F);
    normalize(tmp, tmp, 1, 0, NORM_MINMAX);
    tmp = 255 * tmp;
    tmp.convertTo(tmp, CV_8U, 1, 0);
    return tmp;
}

int Examples::sobelFilter() {
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
    gx = scaleImage2_uchar(gx);
    gy = scaleImage2_uchar(gy);
    g = scaleImage2_uchar(g);
    for(;;) {
        imshow("img", img);
        imshow("gx", gx);
        imshow("gy", gy);
        imshow("g", g);
        if ((char)waitKey(5) == 'q') break;
    }

    return 0;
}

int Examples::fourBlurTypes() {
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

int Examples::easySobelFilter() {
    namedWindow("img", WINDOW_KEEPRATIO);
    namedWindow("gx", WINDOW_KEEPRATIO);
    namedWindow("gy", WINDOW_KEEPRATIO);
    namedWindow("g", WINDOW_KEEPRATIO);

    Mat img = imread("img/lena.png", IMREAD_GRAYSCALE);
    Mat gx, gy, g;
    spatialGradient(img, gx, gy, 3, BORDER_DEFAULT);
    g = abs(gx) + abs(gy);
    gx = this->scaleImage2_uchar(gx);
    gy = this->scaleImage2_uchar(gy);
    g  = this->scaleImage2_uchar(g);

    for (;;) {
        imshow("img", img);
        imshow("gx", gx);
        imshow("gy", gy);
        imshow("g", g);

        if ((char)waitKey(1)=='q') break;
    }

    return 0;
}

Mat Examples::createCosineImg(const int &rows, const int cols, const float &freq, const float &theta) {
    Mat img = Mat::zeros(rows, cols, CV_32F);
    float rho;

    for (int x = 0; x < img.cols; x++)
    {
        for (int y = 0; y < img.rows; y++) {
            rho = x * cos(theta) - y*sin(theta);
            img.at<float>(y, x) = cos(2*CV_PI*freq*rho);
        }
    }

    return img;
}

Mat Examples::createWhiteDisk(const int &rows, const int &cols, const int &cx, const int &cy, const int &radius) {
    Mat disk = Mat::zeros(rows, cols, CV_32F);

    for (int x = 0; x < disk.cols; ++x) {
        for (int y = 0; y < disk.rows; ++y) {
            float d = pow((x-cx) * (x-cx) + (y-cy) * (y-cy), 0.5);
            if (d <= radius)
            {
                //disk.at<float>(x, y) = 1.0;
                disk.at<float>(x, y) = 1 - d / radius;
            }
        }
    }

    return disk;
}

Mat Examples::fftshift(const Mat &src) {
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

int Examples::easySobelFilter2() {
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
        imshow("img", this->scaleImage2_uchar(img));
        imshow("gx", this->scaleImage2_uchar(gx));
        imshow("gy", this->scaleImage2_uchar(gy));
        imshow("g", this->scaleImage2_uchar(g));

        if ((char)waitKey(1)=='q') break;
    }

    return 0;
}

int Examples::laplacianFilter() {
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
        Mat hist = this->computeHistogram1C(img3);
        add(img, -(factor/100.0)*img2, img3, noArray(), CV_8U);
        imshow("img", this->scaleImage2_uchar(img));
        imshow("img2", this->scaleImage2_uchar(img2));
        imshow("img3", this->scaleImage2_uchar(img3));
        imshow("hist", this->scaleImage2_uchar(hist));
        if ((char)waitKey(5) == 'q') break;
    }

    return 0;
}

int Examples::easyLaplacianFilter() {
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
        imshow("lap", this->scaleImage2_uchar(lap));
        if ((char)waitKey(5) == 'q') break;
    }

    return 0;
}



int Examples::dftMethod() {
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
        imshow("img", this->scaleImage2_uchar(img));
        imshow("planes_0", this->fftshift(planes[0]));
        imshow("planes_1", this->fftshift(planes[1]));
        if ((char)waitKey(5) == 'q') break;
    }

    return 0;
}

int Examples::dftMagnitude() {
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
        imshow("img", this->scaleImage2_uchar(img));
        imshow("planes_0", this->fftshift(planes[0]));
        imshow("planes_1", this->fftshift(planes[1]));
        imshow("mag", this->fftshift(this->scaleImage2_uchar(mag)));
        if ((char)waitKey(5) == 'q') break;
    }

    return 0;
}

int Examples::whiteDisk() {
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

        imshow("disk", this->scaleImage2_uchar(disk));
        if ((char)waitKey(5) == 'q') break;
    }

    return 0;
}



int Examples::easyWhiteDisk() {
    namedWindow("disk", WINDOW_KEEPRATIO);


    Mat disk = Mat::zeros(200, 200, CV_32F);

    int xc = 100;
    int yc = 100;
    int radius = 20;

    createTrackbar("xc", "disk", &xc, disk.cols, 0);
    createTrackbar("yc", "disk", &yc, disk.rows, 0);
    createTrackbar("radius", "disk", &radius, disk.cols, 0);

    for(;;) {
        disk = this->createWhiteDisk(200, 200, xc, yc, radius);

        imshow("disk", this->scaleImage2_uchar(disk));
        if ((char)waitKey(5) == 'q') break;
    }

    return 0;
}

int Examples::lowPassFilterDFT() {
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
        Mat mask = this->createWhiteDisk(img2.rows, img2.cols, (int)img2.cols/2, (int)img2.rows/2, radius);
        mask = this->fftshift(mask);

        Mat planes[] =  {Mat_<float>(img), Mat::zeros(img.size(), CV_32F)};

        merge(planes, 2, img2);
        dft(img2, img2);
        split(img2, planes);

        multiply(planes[0], mask, planes[0]);
        multiply(planes[1], mask, planes[1]);
        merge(planes, 2, img2);
        idft(img2, img2, DFT_REAL_OUTPUT);
        img2 = this->fftshift(img2);

        imshow("img", this->scaleImage2_uchar(img));
        imshow("planes_0", planes[0]);
        imshow("planes_1", planes[1]);
        imshow("img2", this->fftshift(this->scaleImage2_uchar(img2)));
        imshow("mask", mask);
        if ((char)waitKey(5) == 'q') break;
    }

    return 0;
}

int Examples::highPassFilterDFT() {
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
        Mat mask = this->createWhiteDisk(img2.rows, img2.cols, (int)img2.cols/2, (int)img2.rows/2, radius);
        mask = this->fftshift(mask);
        mask = 1 - mask;

        Mat planes[] =  {Mat_<float>(img), Mat::zeros(img.size(), CV_32F)};

        merge(planes, 2, img2);
        dft(img2, img2);
        split(img2, planes);

        multiply(planes[0], mask, planes[0]);
        multiply(planes[1], mask, planes[1]);
        merge(planes, 2, img2);
        idft(img2, img2, DFT_REAL_OUTPUT);
        img2 = this->fftshift(img2);

        imshow("img", this->scaleImage2_uchar(img));
        imshow("planes_0", planes[0]);
        imshow("planes_1", planes[1]);
        imshow("img2", this->fftshift(this->scaleImage2_uchar(img2)));
        imshow("mask", mask);
        if ((char)waitKey(5) == 'q') break;
    }

    return 0;
}

int Examples::cosineImage() {
    namedWindow("img", WINDOW_KEEPRATIO);


    Mat img;
    int rows = 500;
    int cols = 500;
    int freq = 1;
    int theta = 2;

    createTrackbar("Freq", "img", &freq, 100, 0, 0);
    createTrackbar("Theta", "img", &theta, 100, 0, 0);

    for(;;) {
        img = this->createCosineImg(rows, cols, (float)freq/1e3, (float)(2*CV_PI*theta/100.0));
        imshow("img", this->scaleImage2_uchar(img));
        if ((char)waitKey(5) == 'q') break;
    }

    return 0;
}

int Examples::cosineNoise() {
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
        noise = this->createCosineImg(rows, cols, (float)freq/1e3, (float)(2*CV_PI*theta/100.0));

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

        imshow("img", this->scaleImage2_uchar(noise));
        imshow("mag", this->fftshift(this->scaleImage2_uchar(mag)));
        if ((char)waitKey(5) == 'q') break;
    }

    return 0;
}
