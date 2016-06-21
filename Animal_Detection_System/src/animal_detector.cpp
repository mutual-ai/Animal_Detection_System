/*
 * animal_detector.cpp
 *
 *  Created on: Oct 1, 2015
 *      Author: evmavrop
 */

#include <fstream>
#include <iostream>
#include <opencv2/core/utility.hpp>
#include "opencv2/cudaobjdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/ml.hpp"
#include <time.h>

#include "caffe/caffe.hpp"
#include "caffe/common.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#define CIFAR10 0 //CIFAR10 = 1, Classify CIFAR10 test batch || CIFAR10 = 0, Classify individual images
#define USE_GPU 1 // Use GPU for the first-stage classification
#define USE_CNN 1 // Use the second-stage classification
#define READ_ONE_FILE 1 //READ_ONE_FILE = 1 read only one file || READ_ONE_FILE = 0 read all the files in the input folder

using namespace cv;
using namespace cv::ml;
using namespace caffe;
using std::string;

String SVMModelPath = "trained-SVMs/SVM-CIFAR10-linear-deer-10000-iter-natural-opencv-3.xml";

String caffeModelPath = " ";
String trainedFilePath = " ";
String meanFilePath = " ";
String labelFilePath = " ";

String input_folder = "/home/evmavrop/animal/datasets/deer_images/images_migration/";
String input_file_name = "deer_3071";
String input_file_type = "png";
String input_file = input_file_name + "." + input_file_type;
String output_folder = "/home/evmavrop/animal/datasets/deer_images/output_images4/";
String SVM_output_file = "SVMoutput"+ input_file_name +".png";
String CNN_output_file = "CNNoutput"+ input_file_name +".png";

int resize_x = 1000;
int resize_y = 1000;

double scale = 1.05;
int nlevels = 100;
int gr_threshold = 4;
double hit_threshold = 0;
int win_stride_width = 8;
int win_stride_height = 8;

typedef std::pair<string, float> Prediction;

class Classifier {
public:
    Classifier(const string& model_file, const string& trained_file, const string& mean_file, const string& label_file);

    std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);

private:
    void SetMean(const string& mean_file);

    std::vector<float> Predict(const cv::Mat& img);

    void WrapInputLayer(std::vector<cv::Mat>* input_channels);

    void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);

private:
    shared_ptr<Net<float> > net_;
    cv::Size input_geometry_;
    int num_channels_;
    cv::Mat mean_;
    std::vector<string> labels_;
};

Classifier::Classifier(const string& model_file, const string& trained_file, const string& mean_file, const string& label_file) {

    Caffe::set_mode(Caffe::GPU);

    /* Load the network. */
    net_.reset(new Net<float>(model_file, TEST));
    net_->CopyTrainedLayersFrom(trained_file);

    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

    Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1) << "Input layer should have 1 or 3 channels.";
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

    /* Load the binaryproto mean file. */
    SetMean(mean_file);

    /* Load labels. */
    std::ifstream labels(label_file.c_str());
    CHECK(labels) << "Unable to open labels file " << label_file;
    string line;
    while (std::getline(labels, line)){
        labels_.push_back(string(line));
    }

    Blob<float>* output_layer = net_->output_blobs()[0];
    CHECK_EQ(labels_.size(), output_layer->channels()) << "Number of labels is different from the output layer dimension.";
}

void read_batch(string filename, vector<Mat> &vec, Mat &label){
    std::ifstream file (filename.c_str(), ios::binary);
    if(file.is_open()){
        int number_of_images = 10000;
        int n_rows = 32;
        int n_cols = 32;
        for(int i = 0; i < number_of_images; ++i){
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
        std::cout << "file is not open" << std::endl;
    }
}


Mat concatenateMat(vector<Mat> &vec, bool color){

    int height = vec[0].rows;
    int width = vec[0].cols;
    Mat res = Mat::zeros(height * width, vec.size(), CV_64FC1);
    if(color){
    Mat res = Mat::zeros(height * width, vec.size(), CV_64FC1);
        for(unsigned int i=0; i<vec.size(); i++){
            Mat img(height, width, CV_64FC3);
            Mat gray(height, width, CV_8UC3);
            cvtColor(vec[i], gray, CV_RGB2BGR);
            vec[i].convertTo(img, CV_64FC3);
            // reshape(int cn, int rows=0), cn is num of channels.
            Mat ptmat = img.reshape(0, height * width);
            Rect roi = cv::Rect(i, 0, ptmat.cols, ptmat.rows);
            Mat subView = res(roi);
            ptmat.copyTo(subView);
        }
    }
    else{
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
    }
    //divide(res, 255.0, res);                                 
    return res;
}


static bool PairCompare(const std::pair<float, int>& lhs, const std::pair<float, int>& rhs) {
    return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
    std::vector<std::pair<float, int> > pairs;

    for (size_t i = 0; i < v.size(); i++){
        pairs.push_back(std::make_pair(v[i], i));
    }

    std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);
    std::vector<int> result;

    for (int i = 0; i < N; i++){
        result.push_back(pairs[i].second);
    }

    return result;
}

/* Return the top N predictions. */
std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N) {
    std::vector<float> output = Predict(img);

    N = std::min<int>(labels_.size(), N);
    std::vector<int> maxN = Argmax(output, N);
    std::vector<Prediction> predictions;
    for (int i = 0; i < N; i++){
        int idx = maxN[i];
        predictions.push_back(std::make_pair(labels_[idx], output[idx]));
    }
 
    return predictions;
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file) {
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

    /* Convert from BlobProto to Blob<float> */
    Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);
    CHECK_EQ(mean_blob.channels(), num_channels_) << "Number of channels of mean file doesn't match input layer.";

    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
    std::vector<cv::Mat> channels;
    float* data = mean_blob.mutable_cpu_data();
    for (int i = 0; i < num_channels_; i++) {
        /* Extract an individual channel. */
        cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
        channels.push_back(channel);
        data += mean_blob.height() * mean_blob.width();
    }

  /* Merge the separate channels into a single image. */
    cv::Mat mean;
    cv::merge(channels, mean);

  /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
    cv::Scalar channel_mean = cv::mean(mean);
    mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

std::vector<float> Classifier::Predict(const cv::Mat& img) {
    Blob<float>* input_layer = net_->input_blobs()[0];
    struct timespec  tv1, tv2;

    input_layer->Reshape(1, num_channels_, input_geometry_.height, input_geometry_.width);
    /* Forward dimension change to all layers. */
    net_->Reshape();

    std::vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels);

    Preprocess(img, &input_channels);

//    clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);

    net_->ForwardPrefilled();

//    clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);
//    std::cout << (double)(tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 + (double)(tv2.tv_sec - tv1.tv_sec) << std::endl;

  /* Copy the output layer to a std::vector */
    Blob<float>* output_layer = net_->output_blobs()[0];
    const float* begin = output_layer->cpu_data();
    const float* end = begin + output_layer->channels();

    return std::vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
    Blob<float>* input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); i++){
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
  }
}

void Classifier::Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels) {
    /* Convert the input image to the input image format of the network. */
    cv::Mat sample;
    if(img.channels() == 3 && num_channels_ == 1){
        cv::cvtColor(img, sample, CV_BGR2GRAY);
    }
    else if(img.channels() == 4 && num_channels_ == 1){
        cv::cvtColor(img, sample, CV_BGRA2GRAY);
    }
    else if(img.channels() == 4 && num_channels_ == 3){
        cv::cvtColor(img, sample, CV_BGRA2BGR);
    }
    else if(img.channels() == 1 && num_channels_ == 3){
        cv::cvtColor(img, sample, CV_GRAY2BGR);
    }
    else{
        sample = img;
    }

    cv::Mat sample_resized;
    if(sample.size() != input_geometry_){
        cv::resize(sample, sample_resized, input_geometry_);
    }
    else{
        sample_resized = sample;
    }

    cv::Mat sample_float;
    if(num_channels_ == 3){
        sample_resized.convertTo(sample_float, CV_32FC3);
    }
    else{
        sample_resized.convertTo(sample_float, CV_32FC1);
    }
    cv::Mat sample_normalized;
    cv::subtract(sample_float, mean_, sample_normalized);

    /* This operation will write the separate BGR planes directly to the
     * input layer of the network because it is wrapped by the cv::Mat
     * objects in input_channels. */
    cv::split(sample_normalized, *input_channels);

    CHECK(reinterpret_cast<float*>(input_channels->at(0).data) == net_->input_blobs()[0]->cpu_data()) << "Input channels are not wrapping the input layer of the network.";
}


int main(int argc,char **argv){

    struct timespec  tv1, tv2;

    Ptr<SVM> svm = Algorithm::load<SVM>(SVMModelPath);

    ::google::InitGoogleLogging(argv[0]);

    string model_file   = "/home/evmavrop/animal/cnn/cifar10_full.prototxt";
    string trained_file = "/home/evmavrop/animal/cnn/cifar10_full_iter_70000.caffemodel.h5";
    string mean_file    = "/home/evmavrop/animal/cnn/mean.binaryproto";
    string label_file   = "/home/evmavrop/animal/cnn/labels.txt";
    Classifier classifier(model_file, trained_file, mean_file, label_file);




#if CIFAR10 
    String filename = "/home/evmavrop/animal/datasets/cifar-10-batches-bin/test_batch.bin";
    vector<Mat> batcht;
    Mat labelt = Mat::zeros(1, 10000, CV_64FC1);
    read_batch(filename, batcht, labelt);
    std::cout << batcht.size() << std::endl;
    Mat mttGray = concatenateMat(batcht,false);
    Mat mttColor = concatenateMat(batcht,true);

    mttGray.convertTo(mttGray, CV_32FC1);

    transpose(mttGray, mttGray);
    transpose(mttColor, mttColor);

    Mat hogMat;
    Mat inputImage(32, 32, CV_8UC1);
    Mat inputImageRGB(32, 32, CV_8UC3); 

    std::vector<float> hogDesc;
    int predicted;
    float CNNprob[batcht.size()];
    memset(CNNprob, 0, sizeof(CNNprob));
    std::ofstream myfileCNN("CNNoutput.txt");
    std::ofstream myfileSVM("SVMoutput.txt");
    Ptr<cuda::HOG> gpu_hog= cuda::HOG::create(Size(32, 32), Size(16, 16), Size(8, 8), Size(8, 8), 9);
    for(int i = 0; i < batcht.size(); i++){

        Mat temp_char;

        Mat submat = mttGray(Rect(0, i, 1024, 1));
        vector<float> gpu_descriptor_vec;

        Mat temp = Mat(32, 32, CV_32FC1, submat.data);
        temp.convertTo(temp_char, CV_8UC1);
        cuda::GpuMat gpu_img, gpu_descriptor_temp;
        gpu_img.upload(temp_char);
        gpu_hog->compute(gpu_img, gpu_descriptor_temp);
        Mat gpu_descriptor(gpu_descriptor_temp);
        gpu_descriptor.copyTo(gpu_descriptor_vec);
        hogMat.push_back(gpu_descriptor);

        Mat sample = hogMat.row(i);
        predicted = (int)svm->predict(sample);

        if (predicted > 0){
            std::vector<Prediction> predictions = classifier.Classify(batcht[i]);
            if(predictions[0].first == "deer"){
                CNNprob[i] = predictions[0].second;
            }
            else{
                CNNprob[i] = predictions[1].second;
            }
        }
        myfileSVM << std::to_string(predicted) << std::endl;
        myfileCNN << std::to_string(CNNprob[i]) << std::endl;
    }
    myfileCNN.close();
    myfileSVM.close();


#else
    Ptr<cuda::HOG> gpu_hog = cuda::HOG::create(Size(32, 32), Size(16, 16), Size(8, 8), Size(8, 8), 9);
    HOGDescriptor cpu_hog(Size(32, 32), Size(16, 16), Size(8, 8), Size(8, 8), 9);

    Mat SV = svm->getSupportVectors();
    Mat alpha, svidx;
    double rho = svm->getDecisionFunction(0, alpha, svidx);

    SV = -1. * SV;

    vector<float> svm_model;
    SV.copyTo(svm_model);
    svm_model.push_back(rho);

    gpu_hog->setSVMDetector(svm_model);
    cpu_hog.setSVMDetector(svm_model);

    vector<Rect> found;
    Mat input_img, img_gray, output_img;
    cuda::GpuMat gpu_img;
    Size win_stride(win_stride_width, win_stride_height);
    int i = 0;

#if READ_ONE_FILE
    input_img = imread(input_folder + input_file, IMREAD_COLOR);
    if(!input_img.data ){
        std::cout <<  "Could not open or find the image " << std::endl ;
        exit(-1);
    }
    if(input_img.channels() > 1){
        cvtColor(input_img, img_gray, COLOR_BGR2GRAY);
        input_img.copyTo(output_img);
    }
    else{
        input_img.copyTo(output_img);
        cvtColor(output_img, output_img, COLOR_GRAY2BGR);
        input_img.copyTo(img_gray);
    }

    for(i = 0; i < 1; i++){
        copyMakeBorder(output_img, output_img, 0, 16, 0, 16, BORDER_CONSTANT, Scalar(0));

#else
    vector<String> filenames;
    String folder = "/home/evmavrop/animal/datasets/deer_images/images";
    glob(folder, filenames);

    int imageCount = filenames.size();


    for(i = 0; i < imageCount; i++){
        copyMakeBorder(output_img, output_img, 0, 16, 0, 16, BORDER_CONSTANT, Scalar(0));
        input_img = imread(filenames[i], IMREAD_COLOR);

        if(! input_img.data ){
            std::cout <<  "Could not open or find the image " << i + 1 << " " << filenames[i] << std::endl ;
            exit(-1);
        }

#endif
        if(input_img.channels() > 1){
            cvtColor(input_img, img_gray, COLOR_BGR2GRAY);
            input_img.copyTo(output_img);
        }
        else{
            input_img.copyTo(output_img);
            cvtColor(output_img, output_img, COLOR_GRAY2BGR);
            input_img.copyTo(img_gray);
        }
#if USE_GPU
        gpu_hog->setNumLevels(nlevels);
        gpu_hog->setHitThreshold(hit_threshold);
        gpu_hog->setWinStride(win_stride);
        gpu_hog->setScaleFactor(scale);
        gpu_hog->setGroupThreshold(gr_threshold);
        cuda::Stream stream;
        gpu_img.upload(img_gray, stream);

        clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);

        gpu_hog->detectMultiScale(gpu_img, found);
#else
        clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);

        cpu_hog.nlevels = nlevels;
        cpu_hog.detectMultiScale(img_gray, found, hit_threshold, win_stride, Size(0, 0), scale, gr_threshold, false);
#endif

        clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);
        std::cout << "Total time of SVM = " << (double)(tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 + (double)(tv2.tv_sec - tv1.tv_sec)  << " seconds" << std::endl;

        for(size_t j = 0; j < found.size(); j++){
            Rect r = found[j];
            rectangle(output_img, r.tl(), r.br(), Scalar(0, 255, 0), 1);
        }

    
          imwrite(output_folder + "SVM_" + std::to_string(i) + ".png", output_img);

#if USE_CNN
        input_img.copyTo(output_img);

        clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);
        std::vector<Prediction> predictions = classifier.Classify(input_img);
        clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);
        std::cout << "Total time of CNN = " << (double)(tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 + (double)(tv2.tv_sec - tv1.tv_sec) << std::endl;

        Mat img;
        bool secondClassification[found.size()];
        std::fill_n(secondClassification, found.size(), false);
        float threshold = 0.9;
        for(size_t j = 0; j < found.size(); j++){
            Rect r = found[j];
            img  = Mat(input_img, r);

            std::vector<Prediction> predictions = classifier.Classify(img);

            if((predictions[0].first == "deer" && predictions[0].second > threshold) || (predictions[1].first == "deer" && predictions[1].second > threshold)){
                secondClassification[j] = true;
            }
        }

        int counter = 0;
        for(size_t j = 0; j < found.size(); j++){
            if(secondClassification[j]){
                Rect r = found[j];
                counter++;
                rectangle(output_img, r.tl(), r.br(), Scalar(0, 255, 0), 1);
            }
        }
        imwrite(output_folder + "CNN_" + std::to_string(i) + ".png", output_img);


#endif

    }

#endif

    return 1;

}



