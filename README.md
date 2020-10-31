# modular-slam
vo.cpp - Contains odometry code.
GroundTruth.txt, Rt.txt - Ground truth and calculated pose information for factor graph optimization
Rt_Corrected.txt - Factor graph optimized poses

Although factor graph optimization is performed separately, the text files are only included for program execution.
Compile: g++ -std=c++11 vo.cpp matrix.cpp matrix.h `pkg-config --libs --cflags opencv` -o vo
Execute: ./vo
