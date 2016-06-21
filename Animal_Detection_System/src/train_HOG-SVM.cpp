/*
 * train_HOG-SVM.cpp
 *
 *  Created on: Oct 1, 2015
 *      Author: evmavrop
 */

#include "opencv2/core.hpp"
#include <opencv2/cudaobjdetect.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"
#include <opencv2/imgcodecs.hpp>
#include <fstream>
#include <iostream>

#define TRAIN_SVM 0 
#define CALC_HOG 1
#define SUBSAMPLING 1
#define ITER 10000 

using namespace cv;
using namespace cv::ml;
using namespace std;


String modelPath = "trained-SVMs/test-"+to_string(ITER)+".xml";


void subsampling(Mat &trainX, Mat &trainY){
    
    vector<Mat> classesX(10);
    
    for(int i = 0; i < trainX.rows; i++){
        classesX[trainY.at<int>(0,i)].push_back(trainX(Rect(0, i, 1024, 1)));
    }
    
    trainX = Mat(classesX[4]);
    trainY = Mat(5000, 1, CV_32SC1);
    trainY = Scalar(4);
    Mat temp = Mat(10,1,CV_32SC1);
    
    for(int i = 0; i < 10; i++){
        temp.at<int>(i,0) = i;
    }
    for(int i = 0; i < 555; i++){
        trainX.push_back(classesX[0](Rect(0, i, 1024, 1)));
        trainY.push_back(temp.at<int>(0,0));
        trainX.push_back(classesX[1](Rect(0, i, 1024, 1)));
        trainY.push_back(temp.at<int>(1,0));
        trainX.push_back(classesX[2](Rect(0, i, 1024, 1)));
        trainY.push_back(temp.at<int>(2,0));
        trainX.push_back(classesX[3](Rect(0, i, 1024, 1)));
        trainY.push_back(temp.at<int>(3,0));
        trainX.push_back(classesX[5](Rect(0, i, 1024, 1)));
        trainY.push_back(temp.at<int>(5,0));
        trainX.push_back(classesX[6](Rect(0, i, 1024, 1)));
        trainY.push_back(temp.at<int>(6,0));
        trainX.push_back(classesX[7](Rect(0, i, 1024, 1)));
        trainY.push_back(temp.at<int>(7,0));
        trainX.push_back(classesX[8](Rect(0, i, 1024, 1)));
        trainY.push_back(temp.at<int>(8,0));
        trainX.push_back(classesX[9](Rect(0, i, 1024, 1)));
        trainY.push_back(temp.at<int>(9,0));
    }
    cout << "TrainX rows = " << trainX.rows << endl;
    temp.release();
}

float evaluate(cv::Mat& predicted, cv::Mat& actual) {
    assert(predicted.rows == actual.rows);
    int t = 0;
    int f = 0;
    for(int i = 0; i < actual.rows; i++) {
        float p = predicted.at<int>(i,0);
        float a = actual.at<int>(i,0);
        if((p >= 0.0 && a >= 0.0) || (p <= 0.0 &&  a <= 0.0)) {
            t++;
        } else {
            f++;
        }
    }
    return (t * 1.0) / (t + f);
}

int positive(cv::Mat& predicted){				//positives found
    int pos = 0;
    for(int i = 0; i < predicted.rows; i++) {
        int p = predicted.at<int>(i,0);
	if(p >= 0.0){
		pos++;
	}
    }

    return pos;
}

int negative(cv::Mat& predicted){				//negatives found
    int neg = 0;
    for(int i = 0; i < predicted.rows; i++) {
       	int p = predicted.at<int>(i,0);
	if(p < 0.0){
		neg++;
	}
    }
    return neg;
}

int truePositive(cv::Mat& predicted, cv::Mat& actual){
   assert(predicted.rows == actual.rows);
   int tPos = 0;
   for(int i = 0; i < actual.rows; i++) {
        int p = predicted.at<int>(i,0);
        int a = actual.at<int>(i,0);
        if(p >= 0.0 && a >= 0.0) {
            tPos++;
        }
    }
    return tPos; 
}

int trueNegative(cv::Mat& predicted, cv::Mat& actual){
   assert(predicted.rows == actual.rows);
   int tNeg= 0;
   for(int i = 0; i < actual.rows; i++) {
        int p = predicted.at<int>(i,0);
        int a = actual.at<int>(i,0);
        if(p < 0.0 && a < 0.0) {
            tNeg++;
        }
    }
    return tNeg; 
}


void read_batch(string filename, vector<Mat> &vec, Mat &label){
    ifstream file (filename.c_str(), ios::binary);
    if (file.is_open())
    {
        int number_of_images = 10000;
        int n_rows = 32;
        int n_cols = 32;
        for(int i = 0; i < number_of_images; i++)
        {
            unsigned char tplabel = 0;
            file.read((char*) &tplabel, sizeof(tplabel));
            vector<Mat> channels;
            Mat fin_img = Mat::zeros(n_rows, n_cols, CV_8UC3);
            for(int ch = 0; ch < 3; ch++){
                Mat tp = Mat::zeros(n_rows, n_cols, CV_8UC1);
                for(int r = 0; r < n_rows; r++){
                    for(int c = 0; c < n_cols; c++){
                        unsigned char temp = 0;
                        file.read((char*) &temp, sizeof(temp));
                        tp.at<uchar>(r, c) = (int) temp;
                    }
                }
                channels.push_back(tp);
            }
            cv::merge(channels, fin_img);
            vec.push_back(fin_img);
            label.at<double>(0, i) = (double)tplabel;
        }
    }
    else{
        cout << "file is not open" << endl;
    }
}

Mat concatenateMat(vector<Mat> &vec){

    int height = vec[0].rows;
    int width = vec[0].cols;
    Mat res = Mat::zeros(height * width, vec.size(), CV_64FC1);
    for(unsigned int i=0; i<vec.size(); i++){
        Mat img(height, width, CV_64FC1);
        Mat gray(height, width, CV_8UC1);
        cvtColor(vec[i], gray, CV_RGB2GRAY);
        gray.convertTo(img, CV_64FC1);
        // reshape(int cn, int rows=0), cn is num of channels.
        Mat ptmat = img.reshape(0, height * width);
        Rect roi = cv::Rect(i, 0, ptmat.cols, ptmat.rows);
        Mat subView = res(roi);
        ptmat.copyTo(subView);
    }
    //divide(res, 255.0, res);					//we skip the normalization step here because of the HOG calculation requires the values of the pixels
    return res;
}

void read_CIFAR10(Mat &trainX, Mat &testX, Mat &trainY, Mat &testY){
    
    String filename;
    filename = "/home/evmavrop/animal/datasets/cifar-10-batches-bin/data_batch_1.bin";
    vector<Mat> batch1;
    Mat label1 = Mat::zeros(1, 10000, CV_64FC1);    
    read_batch(filename, batch1, label1);
    
    filename = "/home/evmavrop/animal/datasets/cifar-10-batches-bin/data_batch_2.bin";
    vector<Mat> batch2;
    Mat label2 = Mat::zeros(1, 10000, CV_64FC1);    
    read_batch(filename, batch2, label2);

    filename = "/home/evmavrop/animal/datasets/cifar-10-batches-bin/data_batch_3.bin";
    vector<Mat> batch3;
    Mat label3 = Mat::zeros(1, 10000, CV_64FC1);    
    read_batch(filename, batch3, label3);

    filename = "/home/evmavrop/animal/datasets/cifar-10-batches-bin/data_batch_4.bin";
    vector<Mat> batch4;
    Mat label4 = Mat::zeros(1, 10000, CV_64FC1);    
    read_batch(filename, batch4, label4);

    filename = "/home/evmavrop/animal/datasets/cifar-10-batches-bin/data_batch_5.bin";
    vector<Mat> batch5;
    Mat label5 = Mat::zeros(1, 10000, CV_64FC1);    
    read_batch(filename, batch5, label5);

    filename = "/home/evmavrop/animal/datasets/cifar-10-batches-bin/test_batch.bin";
    vector<Mat> batcht;
    Mat labelt = Mat::zeros(1, 10000, CV_64FC1);    
    read_batch(filename, batcht, labelt);

    Mat mt1 = concatenateMat(batch1);
    Mat mt2 = concatenateMat(batch2);
    Mat mt3 = concatenateMat(batch3);
    Mat mt4 = concatenateMat(batch4);
    Mat mt5 = concatenateMat(batch5);
    Mat mtt = concatenateMat(batcht);

    Rect roi = cv::Rect(mt1.cols * 0, 0, mt1.cols, trainX.rows);
    Mat subView = trainX(roi);
    mt1.copyTo(subView);
    roi = cv::Rect(label1.cols * 0, 0, label1.cols, 1);
    subView = trainY(roi);
    label1.copyTo(subView);

    roi = cv::Rect(mt1.cols * 1, 0, mt1.cols, trainX.rows);
    subView = trainX(roi);
    mt2.copyTo(subView);
    roi = cv::Rect(label1.cols * 1, 0, label1.cols, 1);
    subView = trainY(roi);
    label2.copyTo(subView);

    roi = cv::Rect(mt1.cols * 2, 0, mt1.cols, trainX.rows);
    subView = trainX(roi);
    mt3.copyTo(subView);
    roi = cv::Rect(label1.cols * 2, 0, label1.cols, 1);
    subView = trainY(roi);
    label3.copyTo(subView);

    roi = cv::Rect(mt1.cols * 3, 0, mt1.cols, trainX.rows);
    subView = trainX(roi);
    mt4.copyTo(subView);
    roi = cv::Rect(label1.cols * 3, 0, label1.cols, 1);
    subView = trainY(roi);
    label4.copyTo(subView);

    roi = cv::Rect(mt1.cols * 4, 0, mt1.cols, trainX.rows);
    subView = trainX(roi);
    mt5.copyTo(subView);
    roi = cv::Rect(label1.cols * 4, 0, label1.cols, 1);
    subView = trainY(roi);
    label5.copyTo(subView);

    mtt.copyTo(testX);
    labelt.copyTo(testY);

}

void svm(cv::Mat& trainingData, cv::Mat& trainingClasses, cv::Mat& testData, cv::Mat& testClasses) {
       
#if TRAIN_SVM

    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, ITER, FLT_EPSILON));  
    Ptr<TrainData> data = TrainData::create(trainingData, ROW_SAMPLE, trainingClasses);

    ParamGrid C_grid, Gamma_grid, P_grid, NU_grid, COEF_grid, DEGREE_grid;
    Gamma_grid.logStep = 0; Gamma_grid.minVal = 1; Gamma_grid.maxVal = 1;
    NU_grid.logStep = 0; NU_grid.minVal = 1; NU_grid.maxVal = 1;
    COEF_grid.logStep = 0; COEF_grid.minVal = 1; COEF_grid.maxVal = 1;
    DEGREE_grid.logStep = 0; DEGREE_grid.minVal = 1; DEGREE_grid.maxVal = 1;

    cout << "SVM parameters initialized" << endl;
    cout << "Training..." << endl;
    svm->trainAuto(data, 10, SVM::getDefaultGrid(SVM::C), Gamma_grid, P_grid, NU_grid, COEF_grid, DEGREE_grid, true);
    cout << "Training is completed" << endl;

    cout << "Saving the SVM model to file "<< modelPath << endl;
    svm->save(modelPath);
    cout << "Save is completed" << endl;
#else 

    cout << "Loading trained SVM" << endl;
    Ptr<SVM> svm = Algorithm::load<SVM>(modelPath);
    cout << modelPath << "SVM successfully loaded" << endl; 

#endif
    cout << "Starting testing of SVM" << endl;

    Mat predicted = Mat(testClasses.rows, 1, CV_32SC1);
    for(int i = 0; i < testData.rows; i++) {
        cv::Mat sample = testData.row(i);
        predicted.at<int>(i, 0) = (int)svm->predict(sample);
    }

    int tp, tn, p, n;
    float rec, f1, prec;

    tp = truePositive(predicted, testClasses);
    tn = trueNegative(predicted, testClasses);
    p = positive(predicted);
    n = negative(predicted);
    prec = (tp * 1.0) / (tp + (p-tp));
    rec = (tp * 1.0) / (tp + (n-tn));
    f1 = 2 * prec * rec / (prec + rec);

    cout << endl << "Test data:" << endl;
    cout << "Accuracy=" << evaluate(predicted, testClasses) << endl;
    cout << "True positives = " << tp << " True negative = " << tn << " Positives = " << p << " Negatives = " << n << endl;
    cout << "Precision = " << prec << endl;
    cout << "Recall = " << rec << endl;
    cout << "F1-score = " << f1 << endl;

    predicted.copyTo(testClasses);
    predicted.release();
    predicted = Mat(trainingClasses.rows, 1, CV_32SC1);

    for(int i = 0; i < trainingData.rows; i++) {
        Mat sample = trainingData.row(i);
        predicted.at<int>(i, 0) = (int)svm->predict(sample);
    }

    predicted.copyTo(trainingClasses);
}

int main()
{
    Mat trainX, testX, originalTestX;
    Mat trainY, testY, originalTestY;
    trainX = Mat::zeros(1024, 50000, CV_64FC1);  
    testX = Mat::zeros(1024, 10000, CV_64FC1);  
    trainY = Mat::zeros(1, 50000, CV_64FC1);  
    testY = Mat::zeros(1, 10000, CV_64FC1);  

    cout << "Loading CIFAR10" << endl;
    read_CIFAR10(trainX, testX, trainY, testY);
    cout << "CIFAR10 dataset successfully loaded" << endl;
    
    trainX.convertTo(trainX, CV_32FC1);
    trainY.convertTo(trainY, CV_32SC1);
    testX.convertTo(testX, CV_32FC1);
    testY.convertTo(testY, CV_32SC1);
    
    transpose(trainX, trainX);
    transpose(testX, testX);
    transpose(trainY, trainY);
    transpose(testY, testY);
    testX.copyTo(originalTestX);
#if SUBSAMPLING
    subsampling(trainX, trainY);
#endif

    for (int i = 0; i < trainY.rows; i++){
        if(trainY.at<int>(0,i) == 4){
	    trainY.at<int>(0,i) = 1;
	}
	else{
	    trainY.at<int>(0,i) = -1;
	}
    }

    for (int i = 0; i < testY.rows; i++){
        if(testY.at<int>(0,i) == 4){
	    testY.at<int>(0,i) = 1;
	}
	else{
	    testY.at<int>(0,i) = -1;
	}
    }

    cout << "labels changed to -1  and 1 \n";
    testY.copyTo(originalTestY);

#if CALC_HOG 
    cout << "Calculating hog descriptor on dataset images \n \n";
    Mat hog_dataset;
    Ptr<cuda::HOG> gpu_hog= cuda::HOG::create(Size(32, 32), Size(16, 16), Size(8, 8), Size(8, 8), 9);

    cout << "Descriptor size = " << gpu_hog->getDescriptorSize() << endl;
    cout << "Descriptor format = " << gpu_hog->getDescriptorFormat() << endl;

    for (int i=0; i<trainX.rows; i++){

	Mat temp_char;
        vector<float> gpu_descriptor_vec;
	
	Mat submat = trainX(Rect(0, i, 1024, 1));
        Mat temp = Mat(32, 32, CV_32FC1, submat.data);
	temp.convertTo(temp_char, CV_8UC1);
	cuda::GpuMat gpu_img, gpu_descriptor_temp;
	gpu_img.upload(temp_char);
	gpu_hog->compute(gpu_img, gpu_descriptor_temp);
	Mat gpu_descriptor(gpu_descriptor_temp);
	gpu_descriptor.copyTo(gpu_descriptor_vec);
	hog_dataset.push_back(gpu_descriptor);
    }
    hog_dataset.copyTo(trainX);
    hog_dataset.release();

    for (int i=0; i<testX.rows; i++){
	Mat temp_char;
        vector<float> gpu_descriptor_vec;
	
	Mat submat = testX(Rect(0, i, 1024, 1));
        Mat temp = Mat(32, 32, CV_32FC1, submat.data);
	temp.convertTo(temp_char, CV_8UC1);
	cuda::GpuMat gpu_img, gpu_descriptor_temp;
	gpu_img.upload(temp_char);
	gpu_hog->compute(gpu_img, gpu_descriptor_temp);
	Mat gpu_descriptor(gpu_descriptor_temp);
	gpu_descriptor.copyTo(gpu_descriptor_vec);
	hog_dataset.push_back(gpu_descriptor);
    }

    hog_dataset.copyTo(testX);

#endif

    cout << "Starting train process \n \n";

    svm(trainX,trainY,testX,testY);

    return 0;
}




