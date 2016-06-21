CXX = icc
#CXX = g++



CXXFLAGS = -g3 -Wall -O0 -c -static  -std=c++0x  -fPIC
FASTFLAGS = -O3 -Wall -c -static -std=c++0x  
LDFLAGS = 

OPENCVLIBS = /home/evmavrop/animal/libs/lib 
OPENCVINCL = /home/evmavrop/animal/libs/include 
THIRDPARTYOCV = /home/evmavrop/animal/libs/share/OpenCV/3rdparty/lib

#OPENCV = -lIlmImf -llibjasper -llibtiff -llibpng -llibjpeg -lopencv_contrib -lopencv_stitching -lopencv_nonfree -lopencv_superres -lopencv_ocl -lopencv_ts -lopencv_videostab -lopencv_gpu -lopencv_photo -lopencv_objdetect -lopencv_legacy -lopencv_video -lopencv_ml -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_imgproc -lopencv_flann -lopencv_core -lzlib -lopencv_imgproc 

#OPENCV = -lopencv_contrib -lopencv_stitching -lopencv_nonfree -lopencv_superres -lopencv_ocl -lopencv_ts -lopencv_videostab -lopencv_gpu -lopencv_photo -lopencv_objdetect -lopencv_legacy -lopencv_video -lopencv_ml -lopencv_calib3d -lopencv_features2d -lopencv_highgui -llibjasper -llibtiff -llibpng -llibjpeg -lopencv_imgproc -lopencv_flann -lopencv_core -lzlib -lswscale -lavutil -lavformat -lavcodec -ldc1394 -lgthread-2.0 -lfreetype -lglib-2.0 -lgobject-2.0 -lfontconfig -lpango-1.0 -lcairo -lgdk_pixbuf-2.0 -lpangocairo-1.0 -lpangoft2-1.0 -lgio-2.0 -latk-1.0 -lgdk-x11-2.0 -lgtk-x11-2.0 -lIlmThread -lHalf -lIex -lIlmImf -lImath -lcufft -lcublas -lrt -lpthread -lm -ldl -lstdc++ -lnpps -lnppi -lnppc -lcudart
 
OPENCV = -lopencv_cudabgsegm -lopencv_cudaobjdetect -lopencv_cudastereo -lopencv_shape -lopencv_stitching -lopencv_cudafeatures2d -lopencv_superres -lopencv_cudacodec -lopencv_videostab -lopencv_cudaoptflow -lopencv_cudalegacy -lopencv_calib3d -lopencv_features2d -lopencv_objdetect -lopencv_highgui -lopencv_videoio -lopencv_photo -lopencv_imgcodecs -lopencv_cudawarping -lopencv_cudaimgproc -lopencv_cudafilters -lopencv_video -lopencv_ml -lopencv_imgproc -lopencv_flann -lopencv_cudaarithm -lopencv_core -lopencv_cudev -lzlib -llibjpeg -llibwebp -llibpng -llibtiff -llibjasper -L/usr/lib/x86_64-linux-gnu -lImath -lIlmImf -lIex -lHalf -lIlmThread -lgtk-x11-2.0 -lgdk-x11-2.0 -latk-1.0 -lgio-2.0 -lpangoft2-1.0 -lpangocairo-1.0 -lgdk_pixbuf-2.0 -lcairo -lpango-1.0 -lfontconfig -lgobject-2.0 -lfreetype -lgthread-2.0 -lglib-2.0 -ldc1394 -lavcodec -lavformat -lavutil -lswscale -lstdc++ -ldl -lm -lpthread -lrt -lcudart -lnppc -lnppi -lnpps -lcublas -lcufft 


THREADS = -pthread

CAFFELIB = /home/evmavrop/animal/caffe/build/lib
CAFFEINCL = /home/evmavrop/animal/caffe/include

CAFFE = -lcaffe

CUDALIB = /usr/local/cuda/lib64
CUDAINCL = /usr/local/cuda/include

CUDA = -lcudart  -lcublas -lcurand

GNULIB = /usr/lib/x86_64-linux-gnu 

BOOST = -lboost_system -lboost_filesystem -lboost_thread

LIBS = /usr/lib 
SRC = /home/evmavrop/animal 
INCL = /home/evmavrop/animal

BLAS_INCLUDE = /opt/intel/mkl/include
BLAS_LIB = /opt/intel/mkl/lib/intel64
MKL = -lmkl_rt

OTHERLIBS = -lstdc++ -lglog -lgflags -lprotobuf -lm -lhdf5_hl -lhdf5

train: trainHOG-SVM

animal: animal_detector

sample: opencv_sample

fast_animal: fast_animal_detector

fast_train: fast_trainHOG-SVM

FASTOBJ = $(addprefix fast_,%.o...)

.PHONY: clean train test animal sample fast_animal fast_train

obj/%.o: src/%.cpp 
	$(CXX) $(CXXFLAGS) -I$(OPENCVINCL) -I$(CAFFEINCL) -I$(CUDAINCL) -I$(BLAS_INCLUDE) $< -o $@

obj/fast_animal_detector.o: src/animal_detector.cpp
	$(CXX) $(FASTFLAGS) -I$(OPENCVINCL) -I$(CAFFEINCL) -I$(CUDAINCL) -I$(BLAS_INCLUDE) $< -o $@

obj/fast_train_HOG-SVM.o: src/train_HOG-SVM.cpp
	$(CXX) $(FASTFLAGS) -I$(OPENCVINCL) -I$(CAFFEINCL) -I$(CUDAINCL) -I$(BLAS_INCLUDE) $< -o $@

trainHOG-SVM: obj/train_HOG-SVM.o
	$(CXX) -L$(OPENCVLIBS) -L$(THIRDPARTYOCV) -L$(CUDALIB) -L$(GNULIB) -o train_hog-svm obj/train_HOG-SVM.o $(OPENCV) $(CUDA) 

animal_detector: obj/animal_detector.o
	$(CXX) -L$(OPENCVLIBS) -L$(THIRDPARTYOCV) -L$(CUDALIB) -L$(GNULIB) -L$(CAFFELIB) -L$(BLAS_LIB) -o animal_detector obj/animal_detector.o  $(OPENCV) $(CUDA) $(CAFFE) $(BOOST) $(MKL) $(OTHERLIBS)

fast_animal_detector: obj/fast_animal_detector.o
	$(CXX) -L$(OPENCVLIBS) -L$(THIRDPARTYOCV) -L$(CUDALIB) -L$(GNULIB) -L$(CAFFELIB) -L$(BLAS_LIB) -o fast_animal_detector obj/fast_animal_detector.o  $(OPENCV) $(CUDA) $(CAFFE) $(BOOST) $(MKL) $(OTHERLIBS)

fast_trainHOG-SVM: obj/fast_train_HOG-SVM.o
	$(CXX) -L$(OPENCVLIBS) -L$(THIRDPARTYOCV) -L$(CUDALIB) -L$(GNULIB) -o fast_train_hog-svm obj/fast_train_HOG-SVM.o $(OPENCV) $(CUDA) 


clean:
	rm -f obj/*.o test train_hog-svm  fast_train_hog-svm fast_animal_detector animal_detector


