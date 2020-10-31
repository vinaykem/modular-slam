# modular-slam
vo.cpp - Contains odometry code.

Rt_Corrected.txt - Factor graph optimized poses. (Although factor graph optimization is performed separately, the text files are only included for program execution.)
Before execution, make sure to change dataset filenames in vo.cpp(Line 22, 151, 152)

Steps to execute:
1. Checkout the main branch
2. mkdir build
3. cmake ..
4. make
5. ./vo
