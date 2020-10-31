#include<opencv2/core.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>
#include<opencv2/xfeatures2d.hpp>
#include<opencv2/features2d.hpp>
#include "opencv2/calib3d.hpp"
#include<sstream>
#include<fstream>
#include "matrix.h"
#include<unistd.h>

using namespace std;
using namespace cv;


double getAbsoluteScale(int frame_id, int sequence_id, double z_cal)	{

  string line;
  int i = 0;
  ifstream myfile ("/shard/KITTI/00/00.txt");
  double x =0, y=0, z = 0;
  double x_prev, y_prev, z_prev;
  if (myfile.is_open())
  {
    while (( getline (myfile,line) ) && (i<=frame_id))
    {
      z_prev = z;
      x_prev = x;
      y_prev = y;
      std::istringstream in(line);
      for (int j=0; j<12; j++)  {
        in >> z ;
        if (j==7) y=z;
        if (j==3)  x=z;
      }

      i++;
    }
    myfile.close();
  }

  else {
    cout << "Unable to open file";
    return 0;
  }

  return sqrt((x-x_prev)*(x-x_prev) + (y-y_prev)*(y-y_prev) + (z-z_prev)*(z-z_prev)) ;

}


// Match keypoints and descriptors.
// Returns: 2D point vectors
void matchKpsAndDes(vector<cv::KeyPoint> kp1, vector<cv::KeyPoint> kp2, Mat d1, Mat d2, vector<Point2f> &point1, vector<Point2f> &point2) {

	cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING2, true);
        vector<cv::DMatch> matches;

        matcher->match(d2, d1, matches);
        sort(matches.begin(), matches.end());
        int goodMatches = matches.size();
        // matches.erase(matches.begin()+goodMatches, matches.end());
        // cout << "Matches:" << matches.size() << endl;
        for(int i = 0; i < matches.size(); i++) {
                point1.push_back(kp1[matches[i].trainIdx].pt);
                point2.push_back(kp2[matches[i].queryIdx].pt);
        }
	// cout << point1.size() << ":Points:" << point2.size() << endl;
}

// Load ground truth poses
// Needed for matching trajectories and for factor graphs
vector<Matrix> loadPoses(string file_name) {
  vector<Matrix> poses;
  FILE *fp = fopen(file_name.c_str(),"r");
  if (!fp)
    return poses;
  while (!feof(fp)) {
    Matrix P = Matrix::eye(4);
    if (fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                   &P.val[0][0], &P.val[0][1], &P.val[0][2], &P.val[0][3],
                   &P.val[1][0], &P.val[1][1], &P.val[1][2], &P.val[1][3],
                   &P.val[2][0], &P.val[2][1], &P.val[2][2], &P.val[2][3] )==12) {
      poses.push_back(P);
      //cout << P.val[0][3] << ":" << P.val[1][3] << ":" << P.val[2][3] << endl;
    }
  }
  fclose(fp);
  return poses;
}


// Heloer function to load optimized rotations and translations
// from factor graphs
vector<vector<double>> loadCorrected(string filename) {
	cout << filename << endl;
	ifstream infile("Rt_Corrected.txt");
	bool rot = false; int count = 3;
	vector<double> rt_corrected;
	vector<vector<double>> all_rt_c;
	string line;
	while(getline(infile, line)) {
		string s;
		//getline(infile, line);
		stringstream ss(line);
		
		double a,b,c;
		if(count < 3) {
			ss >> a >> b >> c;
			rt_corrected.push_back(a);
			rt_corrected.push_back(b);
			rt_corrected.push_back(c);
			count++;
			continue;
		}
		ss >> s;
		if(s == "Value") continue;
		if(s == "R:") {
			count = 0;
			continue;
		}
		if(s == "t:") {
			ss >> a >> b >> c;
			rt_corrected.push_back(a);
                        rt_corrected.push_back(b);
                        rt_corrected.push_back(c);
			all_rt_c.push_back(rt_corrected);
			rt_corrected.clear();
		}

	}
	cout << all_rt_c.size() << endl;
	return all_rt_c;
}

int main()
{

	vector<cv::String> image_path;

	// Pose information during the trajectory course
	Mat t_f, R_f;

	// Optimized pose information obtained from factor graph
	Mat tcorr, Rcorr;
	vector<vector<double>> corrected_rt = loadCorrected("RT_Corrected.txt");
	
	// Dataset location
        glob("/shard/KITTI/00/image_0/*.png", image_path, false);
	string line = "/shard/KITTI/00/00.txt";
	
	vector<cv::KeyPoint> kp1, kp2;
        Mat d1, d2;
	Mat prev_img, img;

	vector<Matrix> groundTruth = loadPoses(line);

	ofstream myfile, rtfile, gtfile;
	myfile.open("results.txt");
	rtfile.open("Rt.txt");
	gtfile.open("GroundTruth.txt");

	// Feature detectors
	// cv::Ptr<cv::xfeatures2d::SIFT> detector = cv::xfeatures2d::SIFT::create();
	// cv::Ptr<cv::AKAZE> detector = cv::AKAZE::create();
	cv::Ptr<cv::ORB> detector = cv::ORB::create(2500, 1.4f, 10, 35, 0, 2, ORB::HARRIS_SCORE, 35, 20);

	// Trajectory course output grid
	Mat traj = Mat::zeros(1500, 1500, CV_8UC3);


	for(int i = 0; i < image_path.size(); i++)
	{
		img =  imread(image_path[i], IMREAD_GRAYSCALE);
			
		// Mat mask;
		// cv::goodFeaturesToTrack(img, corners, 2500, 0.05, 3, mask, 3);
		// cv::KeyPoint::convert(corners, kp2);
                //detector->detectAndCompute(img, mask, kp2, d2, false);
		FAST(img, kp2, 20, true);
		detector->compute(img, kp2, d2);

		// Condition skips recovering pose for the first iteration
		// since there will only be a single image
		if(!kp1.empty() && !kp2.empty()) {

			int inliers = 0;
			Mat R, t, mask;
			vector<Point2f> point1, point2;
			int countIter = 1, edgeThresh = 20;

			// Camera parameters
			Mat K = (Mat_<double>(3,3) << 718.8560, 0, 607.1928, 0, 718.8560, 185.2157, 0, 0, 1 );

			// Executes until good inliers are available. egdeThreshold controls number of keypoints. 
			// Lesser edgeThreshold, higher number of keypoints.
			do {
				
				
				if(countIter > 1) {
					FAST(img, kp2, edgeThresh, true);
                			detector->compute(img, kp2, d2);
					FAST(prev_img, kp1, edgeThresh, true);
                			detector->compute(prev_img, kp1, d1);
					point1.clear(); point2.clear();
					R.release(); t.release(); mask.release();
				}
			
				// Match points and descriptors
				matchKpsAndDes(kp1, kp2, d1, d2, point1, point2);
        		
				Mat E = findEssentialMat(point1, point2, K, RANSAC, 0.999, 1.0);

				// Rotation and Translation
				// Calculate Pose
				// Mat R, t, mask;
        			inliers = recoverPose(E, point1, point2, K, R, t, mask);
        			// cout << "Rotation:" << R << endl;
        			// cout << "Translation:" << t << endl;
				cout << "Inliers:" << inliers << endl;
				countIter++;
				edgeThresh-=2;
			} while(inliers < 500 && edgeThresh >= 0);

			if(countIter >= 2) {
				FAST(img, kp2, 20, true);
                                detector->compute(img, kp2, d2);
			}

			// Get scale information
			double scale = 1.0;
			scale = getAbsoluteScale(i, 0, t.at<double>(2));
			
				
			// Load factor graph optimized rotation and translation matrices 
			Mat Rcr_cur, tcr_cur;
			if(i < corrected_rt.size()) {
                                Rcr_cur = (Mat_<double>(3,3) << corrected_rt[i][0],corrected_rt[i][1],corrected_rt[i][2],corrected_rt[i][3],
						corrected_rt[i][4],corrected_rt[i][5],corrected_rt[i][6],corrected_rt[i][7],corrected_rt[i][8]);
                                tcr_cur = (Mat_<double>(3,1) << corrected_rt[i][9],corrected_rt[i][10],corrected_rt[i][11]);
                        }


			if(t_f.empty() && R_f.empty()) {
			       	t_f = t.clone(); R_f = R.clone(); 
				tcorr = tcr_cur.clone(); Rcorr = Rcr_cur.clone(); continue;
			} 
		       
			if ((scale>0.1)&&(-1*t.at<double>(2) > t.at<double>(0)) && (-1*t.at<double>(2) > t.at<double>(1))) {
				t_f = t_f + scale*(R_f*t);
      				R_f = R*R_f;
				if(i < corrected_rt.size()) {
					tcorr = tcr_cur.clone();
				}
    			}

			int x = int(t_f.at<double>(0)) + 600;
    			int y = -int(t_f.at<double>(2)) + 300;
		

			// Write calculated rotation and translation to a file for factor graph optimization	
			rtfile << R_f.at<double>(0,0) << " " << R_f.at<double>(0,1) << " " << R_f.at<double>(0,2) << " "
                                << R_f.at<double>(1,0) << " " << R_f.at<double>(1,1) << " " << R_f.at<double>(1,2) << " "
                                << R_f.at<double>(2,0) << " " << R_f.at<double>(2,1) << " " << R_f.at<double>(2,2) << " "
                                << t_f.at<double>(0) << " " << t_f.at<double>(1) << " " << t_f.at<double>(2) << endl;


			Matrix gt = groundTruth[i];
			Mat gt_mat = (Mat_<double>(3, 4) << gt.val[0][0], gt.val[0][1], gt.val[0][2], gt.val[0][3], 
					gt.val[1][0], gt.val[1][1], gt.val[1][2], gt.val[1][3], gt.val[2][0], gt.val[2][1], gt.val[2][2], gt.val[2][3]);

			// Write ground truth rotation and translation to a file for factor graph optimization
			gtfile  <<  gt_mat.at<double>(0,0) << " " << gt_mat.at<double>(0,1) << " " << gt_mat.at<double>(0,2) << " "
                                << gt_mat.at<double>(1,0) << " " << gt_mat.at<double>(1,1) << " " << gt_mat.at<double>(1,2) << " "
                                << gt_mat.at<double>(2,0) << " " << gt_mat.at<double>(2,1) << " " << gt_mat.at<double>(2,2) << " "
                                << gt_mat.at<double>(0,3) << " " << gt_mat.at<double>(1,3) << " " << gt_mat.at<double>(2,3) << endl;


			int x_truth = int(gt.val[0][3]) + 600;
                        int y_truth = int(gt.val[2][3]) + 300;

			// If corrected translation matrix are available, plot in trajectory using blue.
			if(i < corrected_rt.size()) {
				int x_cor = int(tcorr.at<double>(0)) + 600;
				int y_cor = -int(tcorr.at<double>(2)) + 300;
				cout << "Co-ordinates x_cor:" << x_cor << " :y_cor:" << y_cor << endl;
    				circle(traj, Point(x_cor, y_cor) ,1, CV_RGB(0,0,255), 2);
			}

			// Plot calculated(red) and ground truth(green) trajectories
			cout << "Co-ordinates x:" << x << " :y:" << y << endl;
			myfile << i << ":Co-ordinates x:" << x << " :y:" << y << endl;

			cout << "Co-ordinates x_truth:" << x_truth << " :y_truth:" << y_truth << endl;
			myfile << i << ":Co-ordinates x_truth:" << x_truth << " :y_truth:" << y_truth << endl;
			myfile << "Keypoints1:" << kp1.size() << " :Keypoints2:" << kp2.size() << endl;
    			
			// Plot
			circle(traj, Point(x, y) ,1, CV_RGB(255,0,0), 2);
			circle(traj, Point(x_truth, y_truth), 1, CV_RGB(0, 255, 0), 2);
			rectangle( traj, Point(10, 30), Point(550, 50), CV_RGB(0,0,0), CV_FILLED);
			imshow( "Road facing camera", img );
    			imshow( "Trajectory", traj );

		}
		
		// Previous image, keypoints and descriptors book keeping. Like a sliding window of 2 images every iteration.
		kp1.assign(kp2.begin(), kp2.end()); d1 = d2.clone(); prev_img = img.clone();
		waitKey(1);
	}
	
	// To avoid trajectory from automatically closing after processing all images
	int hold;
	cin >> hold;
	myfile.close();
	rtfile.close();
	return 0;
}
