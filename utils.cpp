#include "utils.h"

Utils::Utils() {

}

Mat Utils::computeHistogram1C (const Mat &src) {
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

Mat Utils::computeHistogram3C (const Mat &src) {
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

Mat Utils::scaleImage2_uchar(Mat &src) {
    Mat tmp = src.clone();
    if (src.type() != CV_32F)
        tmp.convertTo(tmp, CV_32F);
    normalize(tmp, tmp, 1, 0, NORM_MINMAX);
    tmp = 255 * tmp;
    tmp.convertTo(tmp, CV_8U, 1, 0);
    return tmp;
}

Mat Utils::createCosineImg(const int &rows, const int cols, const float &freq, const float &theta) {
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

Mat Utils::createWhiteDisk(const int &rows, const int &cols, const int &cx, const int &cy, const int &radius) {
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

Mat Utils::fftshift(const Mat &src) {
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

Mat Utils::cvtImg2Colormap(const Mat &src, int colormap) {
    Mat output = src.clone();
    output = this->scaleImage2_uchar(output);
    applyColorMap(output, output, colormap);
    return output;
}
