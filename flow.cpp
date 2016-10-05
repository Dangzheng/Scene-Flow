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

int main(int argc, char *argv[]) {

  if (argc < 3) {
    std::cerr << "usage: ./flow left_0 left_1" << std::endl;
    exit(1);
  }
  std::cout << "今晚要上演的是：一幕沉重的悲剧..." << std::endl;
  std::string leftImageFilename = argv[1];
  std::string leftplusImageFilename = argv[2];
  // SPSFlow部分
  png::image<png::rgb_pixel> leftImage(leftImageFilename);
  png::image<png::rgb_pixel> leftplusImage(leftplusImageFilename);

  SPSFlow flow;
  //首先确认参数设置正确。
  flow.setOutputVZRatio(90);
  flow.setIterationTotal(outerIterationTotal, innerIterationTotal);
  flow.setWeightParameter(lambda_pos, lambda_depth, lambda_bou, lambda_smo);
  flow.setInlierThreshold(lambda_d);
  flow.setPenaltyParameter(lambda_hinge, lambda_occ, lambda_pen);

  png::image<png::gray_pixel_16> segmentImage;
  png::image<png::gray_pixel_16> vzratioImage;
  std::vector<std::vector<double>> vzratioPlaneParameters;
  std::vector<std::vector<int>> boundaryLabels;

  flow.compute(superpixelTotal, leftImage, leftplusImage, segmentImage,
               vzratioImage, vzratioPlaneParameters, boundaryLabels,
               leftImageFilename, leftplusImageFilename);

  return 0;
}
