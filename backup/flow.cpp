//标准库
#include <iostream>
#include <math.h>
#include <string>
#include <vector>
// opencv
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/opencv.hpp>
// png++
#include <png++/png.hpp>
// Flow 类
#include "SPSFlow.h"
#include "defParameter.h"

using namespace cv;

bool interscetion(Point2f &inter_pt, Point2f o1, Point2f p1, Point2f o2,
                  Point2f p2) {
  Point2f x = o2 - o1;
  Point2f d1 = p1 - o1;
  Point2f d2 = p2 - o2;

  float cross = d1.x * d2.y - d1.y * d2.x;
  if (std::abs(cross) < /*EPS*/ 1e-8)
    return false;

  double t1 = (x.x * d2.y - x.y * d2.x) / cross;
  inter_pt = o1 + d1 * t1;
  return true;
};

int main(int argc, char *argv[]) {

  if (argc < 3) {
    std::cerr << "usage: ./flow left_0 left_1" << std::endl;
    exit(1);
  }

  std::string leftImageFilename = argv[1];
  std::string leftplusImageFilename = argv[2];

  Mat leftImage0, leftImage1;
  leftImage0 = imread(leftImageFilename, 1);
  leftImage1 = imread(leftplusImageFilename, 1);

  // imshow("Flow_image_0",leftImage0);
  // imshow("Flow_image_1",leftImage1);
  // waitKey(0);

  // SIFT
  SiftFeatureDetector siftdtc;
  vector<KeyPoint> kp0, kp1;
  Mat Keypoint0, Keypoint1;

  siftdtc.detect(leftImage0, kp0);
  // drawKeypoints(leftImage0, kp0, Keypoint0);
  // imshow("leftImage0_keypoints",Keypoint0);
  // waitKey(0);

  /*
  vector<KeyPoint>::iterator itvc;

  for(itvc=kp0.begin();itvc!=kp0.end();itvc++)
  {
      std::cout<<"angle:"<<itvc->angle<<"\t"<<itvc->class_id<<"\t"<<itvc->octave<<"\t"<<itvc->pt<<"\t"<<itvc->response<<std::endl;
  }
  */
  // angle表示特征点的方向，负值表示不使用
  // class_id表示聚类的ID
  // pt 表示坐标

  siftdtc.detect(leftImage1, kp1);
  // drawKeypoints(leftImage1, kp1, Keypoint1);

  SiftDescriptorExtractor extractor;
  Mat descriptor0, descriptor1;
  BruteForceMatcher<L2<float>> matcher;
  vector<DMatch> matches;
  Mat MatchesImage;
  extractor.compute(leftImage0, kp0, descriptor0);
  extractor.compute(leftImage1, kp1, descriptor1);

  // imshow("descriptor",descriptor0);
  // waitKey(0);

  // std::cout<<std::endl<<descriptor0<<std::endl;
  matcher.match(descriptor0, descriptor1, matches);
  drawMatches(leftImage0, kp0, leftImage0, kp1, matches, MatchesImage);
  vector<int> compression_params;
  compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
  compression_params.push_back(9);
  imwrite("before.png", MatchesImage, compression_params);
  // imshow("Match",MatchesImage);
  // waitKey(0);

  // LMeds或者8-points估计fundamental matrix

  // kp0,kp1为feature points需要将其转换array
  int ptCount = matches.size();
  vector<Point2f> points0(ptCount);
  vector<Point2f> points1(ptCount);
  Point2f pt;
  for (int i = 0; i < ptCount; i++) {
    pt = kp0[matches[i].queryIdx].pt;
    points0[i].x = pt.x;
    points0[i].y = pt.y;
    pt = kp1[matches[i].trainIdx].pt;
    points1[i].x = pt.x;
    points1[i].y = pt.y;
  }

  //根据第一次的计算结果去除outlier和误差大的点。
  Mat FundamentalMatrix;
  vector<uchar> LMedsStatus;
  FundamentalMatrix =
      findFundamentalMat(points0, points1, CV_FM_LMEDS, 3, 0.99, LMedsStatus);
  int Outliers = 0;
  for (int i = 0; i < ptCount; i++) {
    if (LMedsStatus[i] == 0) {
      Outliers++;
    }
  }
  // std::cout << "Outliers:" << Outliers << std::endl;
  // std::cout << "ptCount:" << ptCount << std::endl;
  //根据点的status来去除outlier。

  vector<Point2f> pt0_inlier;
  vector<Point2f> pt1_inlier;
  vector<DMatch> InlierMatches;
  int InlierCount = ptCount - Outliers;
  InlierMatches.resize(InlierCount);
  pt0_inlier.resize(InlierCount);
  pt1_inlier.resize(InlierCount);
  InlierCount = 0;
  for (int i = 0; i < ptCount; i++) {
    if (LMedsStatus[i] != 0) {
      pt0_inlier[InlierCount].x = points0[i].x;
      pt0_inlier[InlierCount].y = points0[i].y;
      pt1_inlier[InlierCount].x = points1[i].x;
      pt1_inlier[InlierCount].y = points1[i].y;
      InlierMatches[InlierCount].queryIdx = InlierCount;
      InlierMatches[InlierCount].trainIdx = InlierCount;
      InlierCount++;
    }
  }

  FundamentalMatrix =
      findFundamentalMat(pt0_inlier, pt1_inlier, CV_FM_LMEDS, 3, 0.99);
  vector<KeyPoint> key0(InlierCount);
  vector<KeyPoint> key1(InlierCount);
  KeyPoint::convert(pt0_inlier, key0);
  KeyPoint::convert(pt1_inlier, key1);
  drawMatches(leftImage0, key0, leftImage0, key1, InlierMatches, MatchesImage);
  imwrite("after.png", MatchesImage, compression_params);
  // imshow("Match",MatchesImage);
  // waitKey(0);
  /*
  FileStorage fs("FundamentalMat.xml", FileStorage::WRITE);
  fs << "fundamentalMat" <<FundamentalMatrix;
  fs.release();
  */

  // CALIBRATION QUALITY CHECK
  // epipolar geometry constraint: m2^t*F*m1=0

  double err = 0;
  int npoints = 0;
  int nimages = 1;
  vector<Vec3f> lines[2];
  vector<vector<Point2f>> imagePoints[2];
  imagePoints[0].resize(nimages);
  imagePoints[1].resize(nimages);
  imagePoints[0][0] = pt0_inlier;
  imagePoints[1][0] = pt1_inlier;
  int err_count = 0;
  vector<Point2f> errpt_0;
  vector<Point2f> errpt_1;
  errpt_0.resize(InlierCount);
  errpt_1.resize(InlierCount);
  for (int i = 0; i < nimages; i++) {
    Mat imgpt[2];
    int npt = (int)imagePoints[0][i].size();

    for (int k = 0; k < 2; k++) {
      imgpt[k] = Mat(imagePoints[k][i]);
      computeCorrespondEpilines(imgpt[k], k + 1, FundamentalMatrix, lines[k]);
      //绘制极线
      /*
      RNG &rng = theRNG();
      if (k == 1) {
        for (int i = 0; i < pt1_inlier.size(); i++) {
          Scalar color = Scalar(rng(256), rng(256), rng(256));
          cv::circle(leftImage1, pt1_inlier[i], 5, color);
          cv::line(leftImage1, Point(0, -lines[k][i][2] / lines[k][i][1]),
                   Point(leftImage1.cols,
                         -(lines[k][i][2] + lines[k][i][0] * leftImage0.cols) /
                             lines[k][i][1]),
                   color);

          cv::imshow("leftImagePlus's epiline", leftImage1);
          waitKey(0);
        }
      }
      */
      //计算右图像的极点，利用的是两条直线的交点，所有的匹配点对两两成对，然后计算出一堆
      //极点的备选点，计算这些点的均值和方差用于滤除outlier
      Mat epipole_mean_x, epipole_mean_y;
      if (k == 1) {
        Point2f inter_pt, o0, o1, p0, p1;
        Mat epipole_mean_x, epipole_var_x, epipole_var_y;
        Mat epipole_x(pt1_inlier.size() - 1, 1, CV_32F);
        Mat epipole_y(pt1_inlier.size() - 1, 1, CV_32F);

        for (int i = 0; i < pt1_inlier.size() - 1; i++) {
          o0.x = 0;
          o0.y = -lines[k][i][2] / lines[k][i][1];
          p0.x = leftImage1.cols;
          p0.y = -(lines[k][i][2] + lines[k][i][0] * leftImage0.cols) /
                 lines[k][i][1];
          o1.x = 0;
          o1.y = -lines[k][i + 1][2] / lines[k][i + 1][1];
          p1.x = leftImage1.cols;
          p1.y = -(lines[k][i + 1][2] + lines[k][i + 1][0] * leftImage0.cols) /
                 lines[k][i + 1][1];
          interscetion(inter_pt, o0, p0, o1, p1);
          epipole_x.at<float>(i, 0) = inter_pt.x;
          epipole_y.at<float>(i, 0) = inter_pt.y;

          /*
          Scalar color = Scalar(rng(256), rng(256), rng(256));
          cv::circle(leftImage1, inter_pt, 5, color);
          imshow("leftImagePlus's epipole", leftImage1);
          waitKey(0);
          std::cout << "cross point is:" << inter_pt.x << "," << inter_pt.y
                    << std::endl;
          */
        }
        meanStdDev(epipole_x, epipole_mean_x, epipole_var_x);
        meanStdDev(epipole_y, epipole_mean_y, epipole_var_y);
        Mat temp_x, temp_y;
        for (int i = 0; i < epipole_x.rows; i++) {

          if (fabs(epipole_x.at<float>(i, 0) -
                   epipole_mean_x.at<double>(0, 0)) < 0.1) {
            temp_x.push_back(epipole_x.at<float>(i, 0));
          }
          if (fabs(epipole_y.at<float>(i, 0) -
                   epipole_mean_y.at<double>(0, 0)) < 0.1) {
            temp_y.push_back(epipole_y.at<float>(i, 0));
          }
        }
        swap(epipole_x, temp_x);
        swap(epipole_y, temp_y);
        meanStdDev(epipole_x, epipole_mean_x, epipole_var_x);
        meanStdDev(epipole_y, epipole_mean_y, epipole_var_y);
        std::cout << "epipole coordinate: (" << epipole_mean_x << ","
                  << epipole_mean_y << ")" << std::endl;
        std::cout << "epipole coordinate variance:" << epipole_var_x << "/"
                  << epipole_var_y << std::endl;

        // test:延极线方向的运动
        // 首先，先找到一个点，然后画出他的极线
        int i_t = 2;
        RNG &rng = theRNG();
        Scalar color = Scalar(rng(256), rng(256), rng(256));
        cv::circle(leftImage1, pt1_inlier[i_t], 5, color, -1);
        cv::line(leftImage1, Point(0, -lines[k][i_t][2] / lines[k][i_t][1]),
                 Point(leftImage1.cols, -(lines[k][i_t][2] +
                                          lines[k][i_t][0] * leftImage0.cols) /
                                            lines[k][i_t][1]),
                 color);
        Point2f epipole_pt;
        epipole_pt.x = epipole_mean_x.at<double>(0, 0);
        epipole_pt.y = epipole_mean_y.at<double>(0, 0);
        color = Scalar(rng(0), rng(256), rng(0));
        cv::circle(leftImage1, epipole_pt, 5, color, -1);

        float x = epipole_pt.x - pt1_inlier[i_t].x;
        float y = epipole_pt.y - pt1_inlier[i_t].y;
        float square_root = sqrt(pow(x, 2) + pow(y, 2));
        x = x / square_root;
        y = y / square_root;
        float d_epipolar = 40;
        x = d_epipolar * x;
        y = d_epipolar * y;
        Point2f incremental;
        incremental.x = pt1_inlier[i_t].x + x;
        incremental.y = pt1_inlier[i_t].y + y;
        cv::circle(leftImage1, incremental, 5, color);
        // cv::imshow("one of the leftImagePlus's epiline", leftImage1);
        // waitKey(0);
      }
    }
    //----------------------------------------------------------------------

    for (int j = 0; j < InlierCount; j++) {
      double errij =
          fabs(imagePoints[0][i][j].x * lines[1][j][0] +
               imagePoints[0][i][j].y * lines[1][j][1] + lines[1][j][2]) +
          fabs(imagePoints[1][i][j].x * lines[0][j][0] +
               imagePoints[1][i][j].y * lines[0][j][1] + lines[0][j][2]);
      // 齐次表示形式。
      err += errij;
      if (0.01 > errij && errij > 0) {
        errpt_0[err_count].x = imagePoints[0][i][j].x;
        errpt_0[err_count].y = imagePoints[0][i][j].y;
        errpt_1[err_count].x = imagePoints[1][i][j].x;
        errpt_1[err_count].y = imagePoints[1][i][j].y;
        /*
        std::cout << "image 0 coordinate:(" << errpt_0[err_count].x << ","
                  << errpt_0[err_count].y << ")" << std::endl;
        std::cout << "image 1 coordinate:(" << errpt_1[err_count].x << ","
                  << errpt_1[err_count].y << ")" << std::endl;
        std::cout << "error:" << errij << std::endl;
        */
        err_count++;
      }
    }
    npoints += npt;
  }
  std::cout << "average reprojection err = " << err / npoints << std::endl;
  std::cout << "err_count = " << err_count << std::endl;
  vector<DMatch> err_Matches;
  err_Matches.resize(err_count);
  for (int i = 0; i < err_count; i++) {
    err_Matches[i].queryIdx = i;
    err_Matches[i].trainIdx = i;
  }
  vector<KeyPoint> err_key0(InlierCount);
  vector<KeyPoint> err_key1(InlierCount);
  KeyPoint::convert(errpt_0, err_key0);
  KeyPoint::convert(errpt_1, err_key1);
  drawMatches(leftImage0, err_key0, leftImage0, err_key1, InlierMatches,
              MatchesImage);
  imwrite("error.png", MatchesImage, compression_params);

  //利用基础矩阵和相机内参求本质矩阵，SVD分解求相机旋转。
  Mat K_t_2, K_t_3;                              //相机的内参矩阵。
  Mat F(FundamentalMatrix.size(), K_t_2.type()); //数据类型转换。
  FundamentalMatrix.convertTo(F, CV_32FC1);

  /*从文件中读取未完成
  std::fstream calib;
  calib.open("calib.txt");
  if (!calib.is_open()){
      std::cout<<"Error when loading calib result...";exit(1);
  }
  char buffer[300];
  while(!calib.eof() ){
      calib.getline(buffer,300);
      std::cout<<buffer<<std::endl;
  }
  calib.close();
  */

  K_t_2 = (Mat_<float>(3, 3) << 721.537700, 0.000000, 609.559300, 0.000000,
           721.537700, 172.854000, 0.000000, 0.000000, 1.000000);
  K_t_3 = (Mat_<float>(3, 3) << 721.537700, 0.000000, 609.559300, 0.000000,
           721.537700, 172.854000, 0.000000, 0.000000, 1.000000);
  //相机中心的坐标

  Mat EssentialMat;
  EssentialMat = K_t_2.t() * F * K_t_2;
  std::cout << "F:" << std::endl << F << std::endl;

  //基础矩阵计算的应该是没有问题，计算秩确实为2.
  Mat U, W, Vt, Rotation;
  SVD::compute(EssentialMat, W, U, Vt);
  // std::cout << "E:" << std::endl << EssentialMat << std::endl;
  // 照计算机视觉中的多视图几何（9.14）公式写的。
  W = (Mat_<float>(3, 3) << 0, -1, 0, 1, 0, 0, 0, 0, 1);
  Rotation = U * W * Vt;
  // std::cout << "Rotation:" << std::endl << Rotation << std::endl;
  Mat col = (Mat_<float>(3, 1) << 0, 0, 1);
  Mat t = U * col;
  std::cout << "t:" << std::endl << t << std::endl;
  Mat vector_Rot;
  Rodrigues(Rotation, vector_Rot);
  std::cout << "Rotation_vector:" << std::endl << vector_Rot << std::endl;

  // SGMFlow部分，cost computation
  png::image<png::rgb_pixel> leftImage(leftImageFilename);
  png::image<png::rgb_pixel> leftplusImage(leftplusImageFilename);

  SPSFlow flow;
  //首先确认参数设置正确。
  flow.setIterationTotal(outerIterationTotal, innerIterationTotal);
  flow.setWeightParameter(lambda_pos, lambda_depth, lambda_bou, lambda_smo);
  flow.setInlierThreshold(lambda_d);
  flow.setPenaltyParameter(lambda_hinge, lambda_occ, lambda_pen);
  // 自己计算的旋转向量，还有极点的坐标。
  // double wx = vector_Rot.at<float>(0, 0);
  // double wy = vector_Rot.at<float>(0, 1);
  // double wz = vector_Rot.at<float>(0, 2);

  png::image<png::gray_pixel_16> segmentImage;
  png::image<png::gray_pixel_16> vzratioImage;
  std::vector<std::vector<double>> vzratioPlaneParameters;
  std::vector<std::vector<int>> boundaryLabels;

  flow.compute(superpixelTotal, leftImage, leftplusImage, segmentImage,
               vzratioImage, vzratioPlaneParameters, boundaryLabels,
               leftImageFilename, leftplusImageFilename);

  return 0;
}
