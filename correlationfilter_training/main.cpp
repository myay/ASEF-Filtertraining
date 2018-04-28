#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

#include "opencv2/core/utility.hpp"
#include "opencv2/imgcodecs.hpp"

#include <stdio.h>
#include "opencv2/core/types.hpp"
#include "../util/imagesequence.h"
// set the size of the training set here

using namespace cv;
using namespace std;

//char arrays for paths
const char* inputPath;
const char* outputPath;
const char* filterPath;
const char* otherPath;

int training_size = 15;

void correlation(Mat& input, Mat& output, Mat& filter);
void convolveInputxFilter(Mat img_in, Mat filter);
void sobel(Mat& input);

int main(int argc, char** argv)
{
    if(argc < 5)
    {
        cout << "Four paths needed, command should be: path/to/inputimages/ path/to/outputimage/ path/to/filterToStore.png path/to/otherInputsNotInTrainingSet";
        inputPath = "python/input/";
        outputPath = "python/output/robot/";
        filterPath = "python/";
        filterPath = "python/";
        //inputPath = "/home/mikail/Uni/PG/OpenCV/OpenCVSandbox/Images/new/";
        //outputPath = "/home/mikail/Uni/PG/OpenCV/OpenCVSandbox/Images/new/";
    }
    else
    {
        inputPath = argv[1];
        outputPath = argv[2];
        filterPath = argv[3];
        otherPath = argv[4];
        if(argc > 5) training_size = atoi(argv[5]);
    }

    cout << "\nInput Path is: " << inputPath << endl;
    cout << "\nOutput Path is: " << outputPath << endl;
    cout << "\nFilter Path is: " << filterPath << endl;
    cout << "\nOther Inputs Path is: " << otherPath << endl;

    // load training images: input images, (desired) output images
    ImageSequence inputSequence(inputPath, "png");
    ImageSequence outputSequence(outputPath, "png");
    ImageSequence otherSequence(otherPath, "png");

    Mat input;
    Mat output;
    Mat filter;
    Mat avgFilter;
    Mat other;

    Mat filterFloat;

    int imgsetSize = min(training_size, inputSequence.size());
    cout << "IMGSETSIZE IS " << imgsetSize << endl;
    // iterate depending on image set size
    // DO TRAINING
    for(int i = 0; i < imgsetSize; i++)
    {
        inputSequence.get(input);
        inputSequence.increment();
        //imshow("Input", input);
        //waitKey(0);

        outputSequence.get(output);
        outputSequence.increment();
        //waitKey(0);

        // convert input image to sobel image
        sobel(input);
        // calculate filter based on sobel image and expected output
        correlation(input, output, filter);

        if (i == 0)
        {
            // first iteration, initialize avgFilter with same specifications as the filter by cloning
            avgFilter = filter.clone();
            avgFilter.convertTo(other, CV_32F);
            //other.convertTo(avgFilter, CV_8U);
            /*
            int maxFinal;
            for(int i = 0; i < other.rows; i++)
               {
                   float* inv = other.ptr<float>(i);
                   unsigned char* inv2 = avgFilter.ptr<unsigned char>(i);
                   // debug to check values in output image
                   for(int j = 0; j < other.cols; j++)
                   {
                       if( (i%100==0) && (j%100==0))
                       {
                           cout << (int)inv[j] << ", maxF " << endl;
                           cout << (int)inv2[j] << ", maxF " << endl;
                       }

                   }
               }
            */

        }
        else
        {
            // add all filters
            filter.convertTo(filterFloat, CV_32F);
            other += (filterFloat);
        }
        imshow("Input", input);
        imshow("Output", output);
        //imshow("iDFT Filter",filter);
        //imshow("avgFilter step", avgFilter);
        //waitKey(0);
    }
    other /= imgsetSize;
    other.convertTo(avgFilter, CV_8U);
    imshow("AVG SHOW", avgFilter);
    //imwrite(filterPath,avgFilter);
    waitKey(0);

    inputSequence.setCurrentIndex(0);
    // convolve training images with calculated filter
    for(int i = 0; i < imgsetSize; i++)
    {
        inputSequence.get(input);
        inputSequence.increment();
        outputSequence.get(output);
        outputSequence.increment();

        //imshow("Output", output);
        sobel(input);
        convolveInputxFilter(input, avgFilter);

        imshow("Input", input);
        waitKey(0);
    }


    // convolve other images with calculated filter
    for(int i = 0; otherSequence.size() ; i++)
    {
        otherSequence.get(other);
        otherSequence.increment();

        sobel(other);
        imshow("Other input (not in training set)", other);
        convolveInputxFilter(other, avgFilter);
        waitKey(0);
    }


    return 0;
}

void correlation(Mat& img_in, Mat& img_out, Mat& finalImage)
{
    // calculate Fourier Transform of sobel image (input)
    Mat padded;                            //expand input image to optimal size
    int m = getOptimalDFTSize( img_in.rows );
    int n = getOptimalDFTSize( img_in.cols ); // on the border add zero values
    copyMakeBorder(img_in, padded, 0, m - img_in.rows, 0, n - img_in.cols, BORDER_CONSTANT, Scalar::all(0));
    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);         // Add to the expanded another plane with zeros
    dft(complexI, complexI);            // this way the result may fit in the source matrix

    // calculate Fourier Transform of output image
    Mat padded1;                            //expand input image to optimal size
    int m1 = getOptimalDFTSize( img_out.rows );
    int n1 = getOptimalDFTSize( img_out.cols ); // on the border add zero values
    copyMakeBorder(img_out, padded1, 0, m1 - img_out.rows, 0, n1 - img_out.cols, BORDER_CONSTANT, Scalar::all(0));
    Mat planes1[] = {Mat_<float>(padded1), Mat::zeros(padded1.size(), CV_32F)};
    Mat complexO;
    merge(planes1, 2, complexO);         // Add to the expanded another plane with zeros
    dft(complexO, complexO);            // this way the result may fit in the source matrix

    // divide in frequency domain to get filter
    for(int i = 0; i < complexO.rows; i++)
    {
        float* Iit = complexI.ptr<float>(i);
        float* Oit = complexO.ptr<float>(i);
        for(int j = 0; j < complexO.cols*2; j+=2)
        {
            complex<float> a(Oit[j], Oit[j+1]);
            complex<float> b(Iit[j], Iit[j+1]);

            complex<float> c = a / b;

            Oit[j] = c.real();
            Oit[j+1] = c.imag();
        }
    }

    //cout << "\n***iDFT of Filter***\n" << endl;
    Mat inverseTransform;
    dft(complexO, inverseTransform, cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);
    normalize(inverseTransform, inverseTransform, 0, 255, CV_MINMAX);

    int maxFinal = 0;
    inverseTransform.convertTo(finalImage, CV_8U);

    /* //convolution of input with newly attained filter, to get outputimage again
    mulSpectrums(complexI, complexO, complexI, 0);

    Mat fin;
    dft(complexI, fin, cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);
    normalize(fin, fin, 0, 255, CV_MINMAX);
    Mat finalImageConv;
    fin.convertTo(finalImageConv, CV_8U);
    imshow("Input conv filter", finalImageConv);
    */
}

void convolveInputxFilter(Mat img_in, Mat filter)
{
    // calculate Fourier Transform of sobel image (input)
    Mat padded;                            //expand input image to optimal size
    int m = getOptimalDFTSize( img_in.rows );
    int n = getOptimalDFTSize( img_in.cols ); // on the border add zero values
    copyMakeBorder(img_in, padded, 0, m - img_in.rows, 0, n - img_in.cols, BORDER_CONSTANT, Scalar::all(0));
    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);         // Add to the expanded another plane with zeros
    dft(complexI, complexI);            // this way the result may fit in the source matrix

    // calculate Fourier Transform of filter
    Mat padded1;                            //expand input image to optimal size
    int m1 = getOptimalDFTSize( filter.rows );
    int n1 = getOptimalDFTSize( filter.cols ); // on the border add zero values
    copyMakeBorder(filter, padded1, 0, m1 - filter.rows, 0, n1 - filter.cols, BORDER_CONSTANT, Scalar::all(0));
    Mat planes1[] = {Mat_<float>(padded1), Mat::zeros(padded1.size(), CV_32F)};
    Mat complexO;
    merge(planes1, 2, complexO);         // Add to the expanded another plane with zeros
    dft(complexO, complexO);            // this way the result may fit in the source matrix

    mulSpectrums(complexI, complexO, complexO, 0);

    //cout << "\n***iDFT of Output***\n" << endl;
    Mat inverseTransform;
    Mat final;
    dft(complexO, inverseTransform, cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);
    normalize(inverseTransform, inverseTransform, 0, 255, CV_MINMAX);

    int maxFinal = 0;
    inverseTransform.convertTo(final, CV_8U);
    imshow("InputxFilter", final);
}

void sobel(Mat& img_in)
{
    // SOBEL
    Mat sobel;
    Mat grad;

    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;

    // Remove noise by blurring with a Gaussian filter ( kernel size = 3 )
    GaussianBlur(img_in, sobel, Size(5, 5), 0, 0, BORDER_DEFAULT);

    // Sobel to get edges
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;

    Sobel( sobel, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
    Sobel( sobel, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_x, abs_grad_x );
    convertScaleAbs( grad_y, abs_grad_y );
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

    img_in = grad.clone();
    //imshow("Sobel", img_in);
}
