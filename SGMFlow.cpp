#include "SGMFlow.h"
#include "RobustMatcher.h"
#include <algorithm>
#include <math.h>
#include <nmmintrin.h>
#include <stack>
#include <stdexcept>

#include <vector>

// #include <iomanip> 控制输出精度时候使用
// 包含下面这个头文件可以使得_mm_popcnt_u32，在代码美化的时候不报错，
// 但是编译通过不了，所以我换了一个不受平台限制的。
// #include <popcntintrin.h>

// Default parameters
const int SGMFLOW_DEFAULT_VZRATIO_TOTAL = 256;
const double SGMFLOW_DEFAULT_VZRATIO_FACTOR = 256;
const int SGMFLOW_DEFAULT_SOBEL_CAP_VALUE = 15;
const int SGMFLOW_DEFAULT_CENSUS_WINDOW_RADIUS = 2;
const double SGMFLOW_DEFAULT_CENSUS_WEIGHT_FACTOR = 0.7;
const int SGMFLOW_DEFAULT_AGGREGATION_WINDOW_RADIUS = 2;
const int SGMFLOW_DEFAULT_SMOOTHNESS_PENALTY_SMALL = 105;
const int SGMFLOW_DEFAULT_SMOOTHNESS_PENALTY_LARGE = 6450;
const int SGMFLOW_DEFAULT_CONSISTENCY_THRESHOLD = 3;

SGMFlow::SGMFlow()
    : vzratioTotal_(SGMFLOW_DEFAULT_VZRATIO_TOTAL),
      vzratioFactor_(SGMFLOW_DEFAULT_VZRATIO_FACTOR),
      sobelCapValue_(SGMFLOW_DEFAULT_SOBEL_CAP_VALUE),
      censusWindowRadius_(SGMFLOW_DEFAULT_CENSUS_WINDOW_RADIUS),
      censusWeightFactor_(SGMFLOW_DEFAULT_CENSUS_WEIGHT_FACTOR),
      aggregationWindowRadius_(SGMFLOW_DEFAULT_AGGREGATION_WINDOW_RADIUS),
      smoothnessPenaltySmall_(SGMFLOW_DEFAULT_SMOOTHNESS_PENALTY_SMALL),
      smoothnessPenaltyLarge_(SGMFLOW_DEFAULT_SMOOTHNESS_PENALTY_LARGE),
      consistencyThreshold_(SGMFLOW_DEFAULT_CONSISTENCY_THRESHOLD) {}
//添加旋转矩阵的设置参数的类。
void SGMFlow::setVZRatioTotal(const int vzratioTotal) {
  if (vzratioTotal <= 0 || vzratioTotal % 16 != 0) {
    throw std::invalid_argument("[SGMFlow::setVZRatioTotal] the number of "
                                "disparities must be a multiple of 16");
  }

  vzratioTotal_ = vzratioTotal;
}

void SGMFlow::setVZRatioFactor(const double vzratioFactor) {
  if (vzratioFactor <= 0) {
    throw std::invalid_argument("[SGMFlow::setOutputVZRatioFactor] "
                                "vzratio factor is less than zero");
  }

  vzratioFactor_ = vzratioFactor;
}

void SGMFlow::setDataCostParameters(const int sobelCapValue,
                                    const int censusWindowRadius,
                                    const double censusWeightFactor,
                                    const int aggregationWindowRadius) {
  sobelCapValue_ = std::max(sobelCapValue, 15);
  sobelCapValue_ = std::min(sobelCapValue_, 127) | 1;

  if (censusWindowRadius < 1 || censusWindowRadius > 2) {
    throw std::invalid_argument("[SGMFlow::setDataCostParameters] window "
                                "radius of Census transform must be 1 or 2");
  }
  censusWindowRadius_ = censusWindowRadius;
  if (censusWeightFactor < 0) {
    throw std::invalid_argument("[SGMFlow::setDataCostParameters] weight of "
                                "Census transform must be positive");
  }
  censusWeightFactor_ = censusWeightFactor;

  aggregationWindowRadius_ = aggregationWindowRadius;
}

void SGMFlow::setSmoothnessCostParameters(const int smoothnessPenaltySmall,
                                          const int smoothnessPenaltyLarge) {
  if (smoothnessPenaltySmall < 0 || smoothnessPenaltyLarge < 0) {
    throw std::invalid_argument("[SGMFlow::setSmoothnessCostParameters] "
                                "smoothness penalty value is less than zero");
  }
  if (smoothnessPenaltySmall >= smoothnessPenaltyLarge) {
    throw std::invalid_argument("[SGMFlow::setSmoothnessCostParameters] "
                                "small value of smoothness penalty must be "
                                "smaller than large penalty value");
  }

  smoothnessPenaltySmall_ = smoothnessPenaltySmall;
  smoothnessPenaltyLarge_ = smoothnessPenaltyLarge;
}

void SGMFlow::setConsistencyThreshold(const int consistencyThreshold) {
  if (consistencyThreshold < 0) {
    throw std::invalid_argument("[SGMFlow::setConsistencyThreshold] "
                                "threshold for LR consistency must be "
                                "positive");
  }
  consistencyThreshold_ = consistencyThreshold;
}

void SGMFlow::compute(const png::image<png::rgb_pixel> &leftImage,
                      const png::image<png::rgb_pixel> &leftplusImage,
                      float *vzratioImage, std::string leftImageFilename,
                      std::string leftplusImageFilename) {
  initialize(leftImage, leftplusImage);
  calcEpipoleRotaionVector(leftImageFilename, leftplusImageFilename);
  computeCostImage(leftImage, leftplusImage);

  unsigned short *leftVZRatioImage = reinterpret_cast<unsigned short *>(
      malloc(width_ * height_ * sizeof(unsigned short)));
  performSGM(leftCostImage_, leftVZRatioImage);
  unsigned short *leftplusVZRatioImage = reinterpret_cast<unsigned short *>(
      malloc(width_ * height_ * sizeof(unsigned short)));
  png::image<png::gray_pixel> image(width_, height_);
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      vzratioImage[width_ * y + x] =
          static_cast<float>(leftVZRatioImage[width_ * y + x] / vzratioFactor_);
      // outputImage.at<float>(y, x) = vzratioImage[width_ * y + x];
      image[y][x] = vzratioImage[width_ * y + x];
    }
  }
  // performSGM(leftplusCostImage_, leftplusVZRatioImage);
  // enforceLeftRightConsistency(leftVZRatioImage, leftplusVZRatioImage);
  // std::cout << "今晚要上演的是：一幕失落的世界..." << std::endl;
  //
  // png::image<png::gray_pixel> image(width_, height_);
  // for (int y = 0; y < height_; ++y) {
  //   for (int x = 0; x < width_; ++x) {
  //     vzratioImage[width_ * y + x] =
  //         static_cast<float>(leftVZRatioImage[width_ * y + x] /
  //         vzratioFactor_);
  //     // outputImage.at<float>(y, x) = vzratioImage[width_ * y + x];
  //     image[y][x] = vzratioImage[width_ * y + x];
  //   }
  // }

  // cv::imshow("OutputVZratioImage", outputImage);
  image.write("out.png");
  std::cout << "the output file has been written..." << std::endl;
  freeDataBuffer();
  free(leftVZRatioImage);
  free(leftplusVZRatioImage);
}

void SGMFlow::initialize(const png::image<png::rgb_pixel> &leftImage,
                         const png::image<png::rgb_pixel> &leftplusImage) {
  setImageSize(leftImage, leftplusImage); //检测一下输入图像是否
  allocateDataBuffer();
}

void SGMFlow::setImageSize(const png::image<png::rgb_pixel> &leftImage,
                           const png::image<png::rgb_pixel> &leftplusImage) {
  width_ = static_cast<int>(leftImage.get_width());
  height_ = static_cast<int>(leftImage.get_height());
  if (leftplusImage.get_width() != width_ ||
      leftplusImage.get_height() != height_) {
    throw std::invalid_argument("[SGMFlow::setImageSize] sizes of left and "
                                "leftplus images are different");
  }
  widthStep_ = width_ + 15 - (width_ - 1) % 16;
}

void SGMFlow::allocateDataBuffer() {
  leftCostImage_ = reinterpret_cast<unsigned short *>(_mm_malloc(
      width_ * height_ * vzratioTotal_ * sizeof(unsigned short), 16));
  leftplusCostImage_ = reinterpret_cast<unsigned short *>(_mm_malloc(
      width_ * height_ * vzratioTotal_ * sizeof(unsigned short), 16));

  int pixelwiseCostRowBufferSize = width_ * vzratioTotal_;
  int rowAggregatedCostBufferSize =
      width_ * vzratioTotal_ * (aggregationWindowRadius_ * 2 + 2);

  pixelwiseCostRow_ = reinterpret_cast<unsigned char *>(
      _mm_malloc(pixelwiseCostRowBufferSize * sizeof(unsigned char), 16));
  rowAggregatedCost_ = reinterpret_cast<unsigned short *>(
      _mm_malloc(rowAggregatedCostBufferSize * sizeof(unsigned short), 16));

  pathRowBufferTotal_ = 2;
  vzratioSize_ = vzratioTotal_ + 16;
  pathTotal_ = 8; //此处pathtotal指的是在八个方向保持最小，对应SGM那一篇文章。
  pathVZratioSize_ = pathTotal_ * vzratioSize_;

  costSumBufferRowSize_ = width_ * vzratioTotal_;
  costSumBufferSize_ = costSumBufferRowSize_ * height_;
  pathMinCostBufferSize_ = (width_ + 2) * pathTotal_;
  pathCostBufferSize_ = pathMinCostBufferSize_ * vzratioSize_;
  totalBufferSize_ =
      (pathMinCostBufferSize_ + pathCostBufferSize_) * pathRowBufferTotal_ +
      costSumBufferSize_ + 16;

  sgmBuffer_ = reinterpret_cast<short *>(
      _mm_malloc(totalBufferSize_ * sizeof(short), 16));
}

void SGMFlow::freeDataBuffer() {
  _mm_free(leftCostImage_);
  _mm_free(leftplusCostImage_);
  _mm_free(pixelwiseCostRow_);
  _mm_free(rowAggregatedCost_);
  _mm_free(sgmBuffer_);
}

void SGMFlow::calcEpipoleRotaionVector(std::string leftImageFilename,
                                       std::string leftplusImageFilename) {
  cv::Mat leftImage, leftplusImage;
  leftImage = cv::imread(leftImageFilename, 1);
  leftplusImage = cv::imread(leftplusImageFilename, 1);
  // cv::imshow("Flow_image_left", leftImage);
  // cv::imshow("Flow_image_leftplus", leftplusImage);
  // SURF
  RobustMatcher rmatcher;
  rmatcher.setConfidenceLevel(0.99);
  rmatcher.setMinDistanceToEpipolar(0.1);
  rmatcher.setRatio(0.65f);
  cv::Ptr<cv::FeatureDetector> pfd = new cv::SurfFeatureDetector(10);
  rmatcher.setFeatureDetector(pfd);

  std::vector<cv::DMatch> matches;
  std::vector<cv::KeyPoint> keypointsL, keypointsLP;
  cv::Mat fundamental = rmatcher.match(leftImage, leftplusImage, matches,
                                       keypointsL, keypointsLP);
  epipoleX = rmatcher.x();
  epipoleY = rmatcher.y();

  // //在右图中绘制极线
  // int ptCount = matches.size();
  // cv::vector<cv::Point2f> selPointsL(ptCount);
  // cv::vector<cv::Point2f> selPointsLP(ptCount);
  // cv::Point2f pt;
  // for (int i = 0; i < ptCount; i++) {
  //   pt = keypointsLP[matches[i].queryIdx].pt;
  //   selPointsL[i].x = pt.x;
  //   selPointsL[i].y = pt.y;
  //   pt = keypointsLP[matches[i].trainIdx].pt;
  //   selPointsLP[i].x = pt.x;
  //   selPointsLP[i].y = pt.y;
  // }
  // // draw epipoline
  // //
  // std::vector<cv::Vec3f> lines1;
  // cv::computeCorrespondEpilines(cv::Mat(selPointsL), 1, fundamental, lines1);
  // for (std::vector<cv::Vec3f>::const_iterator it = lines1.begin();
  //      it != lines1.end(); ++it) {
  //   cv::line(leftplusImage, cv::Point(0, -(*it)[2] / (*it)[1]),
  //            cv::Point(leftplusImage.cols,
  //                      -((*it)[2] + (*it)[0] * leftplusImage.cols) /
  //                      (*it)[1]),
  //            cv::Scalar(200, 0, 0));
  // }
  // cv::imshow("leftplus", leftplusImage);

  cv::Mat R;
  cv::Mat K(3, 3, CV_64F);
  cv::Mat W(3, 3, CV_64F);
  // KITTI2015
  // K = (cv::Mat_<double>(3, 3) << 721.537700, 0.000000, 609.559300, 0.000000,
  //      721.537700, 172.854000, 0.000000, 0.000000, 1.000000);
  // KITTI2012
  K = (cv::Mat_<double>(3, 3) << 707.091200, 0.000000, 601.887300, 0.000000,
       707.091200, 183.110400, 0.000000, 0.000000, 1.000000);
  // K = (cv::Mat_<double>(3, 3) << 718.856000, 0.000000, 607.192800, 0.000000,
  //      718.856000, 185.215700, 0.000000, 0.000000, 1.000000);
  cv::SVD svd(K.t() * fundamental * K, cv::SVD::MODIFY_A);
  W = (cv::Mat_<double>(3, 3) << 0, -1, 0, 1, 0, 0, 0, 0, 1);
  R = svd.u * W * svd.vt;
  std::cout << "fundamental:" << std::endl << fundamental << std::endl;
  std::cout << "rotation:" << std::endl << R << std::endl;
  // cv::Mat t = svd.u.col(2);
  // std::cout << "t:" << std::endl << t << std::endl;
  cv::Mat vector_Rot, vector_Rot_Inverse;
  cv::Rodrigues(R, vector_Rot);
  cv::Rodrigues(R.t(), vector_Rot_Inverse);
  std::cout << "rotation_vector:" << std::endl << vector_Rot << std::endl;

  wx_t = vector_Rot.at<double>(0, 0);
  wy_t = vector_Rot.at<double>(0, 1);
  wz_t = vector_Rot.at<double>(0, 2);
  std::cout << wx_t << std::endl;
  std::cout << wy_t << std::endl;
  std::cout << wz_t << std::endl;
  // t+1 -> t
  wx_inv = vector_Rot_Inverse.at<double>(0, 0);
  wy_inv = vector_Rot_Inverse.at<double>(0, 1);
  wz_inv = vector_Rot_Inverse.at<double>(0, 2);
}

void SGMFlow::computeCostImage(
    const png::image<png::rgb_pixel> &leftImage,
    const png::image<png::rgb_pixel> &leftplusImage) {
  unsigned char *leftGrayscaleImage = reinterpret_cast<unsigned char *>(
      malloc(width_ * height_ * sizeof(unsigned char)));
  unsigned char *leftplusGrayscaleImage = reinterpret_cast<unsigned char *>(
      malloc(width_ * height_ * sizeof(unsigned char)));
  convertToGrayscale(leftImage, leftplusImage, leftGrayscaleImage,
                     leftplusGrayscaleImage);
  memset(leftCostImage_, 0,
         width_ * height_ * vzratioTotal_ * sizeof(unsigned short));
  std::cout << "starting calc cost..." << std::endl;
  computeLeftCostImage(leftGrayscaleImage, leftplusGrayscaleImage);
  // computeLeftPlusCostImage();

  free(leftGrayscaleImage);
  free(leftplusGrayscaleImage);
}

void SGMFlow::convertToGrayscale(
    const png::image<png::rgb_pixel> &leftImage,
    const png::image<png::rgb_pixel> &leftplusImage,
    unsigned char *leftGrayscaleImage,
    unsigned char *leftplusGrayscaleImage) const {

  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      png::rgb_pixel pix = leftImage.get_pixel(x, y);
      leftGrayscaleImage[width_ * y + x] = static_cast<unsigned char>(
          0.299 * pix.red + 0.587 * pix.green + 0.114 * pix.blue + 0.5);
      pix = leftplusImage.get_pixel(x, y);
      leftplusGrayscaleImage[width_ * y + x] = static_cast<unsigned char>(
          0.299 * pix.red + 0.587 * pix.green + 0.114 * pix.blue + 0.5);
    }
  }
  //
  // for (int y = 0; y < height_; ++y) {
  //   for (int x = 0; x < width_; ++x) {
  //     png::rgb_pixel pix = leftImage.get_pixel(x, y);
  //     leftGrayscaleImage[width_ * y + x] = static_cast<unsigned
  //     char>(pix.red);
  //     pix = leftplusImage.get_pixel(x, y);
  //     leftplusGrayscaleImage[width_ * y + x] =
  //         static_cast<unsigned char>(pix.red);
  //   }
  // }
  //
  //
  // png::image<png::gray_pixel> image(width_, height_);
  // for (int y = 0; y < height_; ++y) {
  //   for (int x = 0; x < width_; ++x) {
  //     leftGrayscaleImage[width_ * y + x] =
  //         static_cast<float>(leftGrayscaleImage[width_ * y + x]);
  //     // outputImage.at<float>(y, x) = vzratioImage[width_ * y + x];
  //     image[y][x] = leftGrayscaleImage[width_ * y + x];
  //   }
  // }
  // image.write("gray.png");
  // std::cout << "the gray file has been written..." << std::endl;
}

void SGMFlow::computeLeftCostImage(
    const unsigned char *leftGrayscaleImage,
    const unsigned char *leftplusGrayscaleImage) {
  // leftSobelImage等原来在此处声明和释放。
  leftSobelImage = reinterpret_cast<unsigned char *>(
      _mm_malloc(widthStep_ * height_ * sizeof(unsigned char), 16));
  leftplusSobelImage = reinterpret_cast<unsigned char *>(
      _mm_malloc(widthStep_ * height_ * sizeof(unsigned char), 16));

  computeCappedSobelIamge(leftGrayscaleImage, false, leftSobelImage);
  computeCappedSobelIamge(leftplusGrayscaleImage, true, leftplusSobelImage);
//   png::image<png::gray_pixel> image(width_, height_);
//   for (int y = 0; y < height_; ++y) {
//     for (int x = 0; x < width_; ++x) {
//       // image[y][x] = leftplusSobelImage[widthStep_ * y + x];
//       image[y][x] = leftplusSobelImage[widthStep_ * y + width_ - 1 - x];
//       //这样才能对上。
//     }
//   }
//   image.write("sobel.png");
//    cv::waitKey(0);

  leftCensusImage =
      reinterpret_cast<int *>(malloc(width_ * height_ * sizeof(int)));
  leftplusCensusImage =
      reinterpret_cast<int *>(malloc(width_ * height_ * sizeof(int)));
  computeCensusImage(leftGrayscaleImage, leftCensusImage);
  computeCensusImage(leftplusGrayscaleImage, leftplusCensusImage);
  //仅仅利用灰度图进行census计算。
  unsigned char *leftSobelRow = leftSobelImage;
  unsigned char *leftplusSobelRow = leftplusSobelImage;
  int *leftCensusRow = leftCensusImage;
  int *leftplusCensusRow = leftplusCensusImage;
  unsigned short *costImageRow = leftCostImage_;

  calcTopRowCost(leftSobelRow, leftCensusRow, leftplusSobelRow,
                 leftplusCensusRow, costImageRow, leftSobelImage,
                 leftplusSobelImage, leftCensusImage, leftplusCensusImage);
  costImageRow += width_ * vzratioTotal_;
  calcRowCosts(leftSobelRow, leftCensusRow, leftplusSobelRow, leftplusCensusRow,
               costImageRow, leftSobelImage, leftplusSobelImage,
               leftCensusImage, leftplusCensusImage);
  //将sobel处理过的图片，census
  // Image送到函数里面计��������配���������������������������������������������������������������������������������������������
}

void SGMFlow::computeCappedSobelIamge(const unsigned char *image,
                                      const bool horizontalFlip,
                                      unsigned char *sobelImage) const {
  memset(sobelImage, sobelCapValue_, widthStep_ * height_);
  //将sobelImage所���内存的前widthStep_ *
  // height_个字节，用sobelCapValue_�����值���进行替换。
  if (horizontalFlip) {
    for (int y = 1; y < height_ - 1; ++y) {
      for (int x = 1; x < width_ - 1; ++x) {
        int sobelValue =
            (image[width_ * (y - 1) + x + 1] + 2 * image[width_ * y + x + 1] +
             image[width_ * (y + 1) + x + 1]) -
            (image[width_ * (y - 1) + x - 1] + 2 * image[width_ * y + x - 1] +
             image[width_ * (y + 1) + x - 1]);
        if (sobelValue > sobelCapValue_)
          sobelValue = 2 * sobelCapValue_;
        else if (sobelValue < -sobelCapValue_)
          sobelValue = 0;
        else
          sobelValue += sobelCapValue_;
        sobelImage[widthStep_ * y + width_ - x - 1] = sobelValue;
      }
    }
  } else {
    for (int y = 1; y < height_ - 1; ++y) {
      for (int x = 1; x < width_ - 1; ++x) {
        int sobelValue =
            (image[width_ * (y - 1) + x + 1] + 2 * image[width_ * y + x + 1] +
             image[width_ * (y + 1) + x + 1]) -
            (image[width_ * (y - 1) + x - 1] + 2 * image[width_ * y + x - 1] +
             image[width_ * (y + 1) + x - 1]);
        if (sobelValue > sobelCapValue_)
          sobelValue = 2 * sobelCapValue_;
        else if (sobelValue < -sobelCapValue_)
          sobelValue = 0;
        else
          sobelValue += sobelCapValue_;
        sobelImage[widthStep_ * y + x] = sobelValue;
      }
    }
  }
}

void SGMFlow::computeCensusImage(const unsigned char *image,
                                 int *censusImage) const {
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      unsigned char centerValue = image[width_ * y + x];

      int censusCode = 0;
      for (int offsetY = -censusWindowRadius_; offsetY <= censusWindowRadius_;
           ++offsetY) {
        for (int offsetX = -censusWindowRadius_; offsetX <= censusWindowRadius_;
             ++offsetX) {
          censusCode = censusCode << 1; //将censusCode的二进制数向左移1位；
          if (y + offsetY >= 0 && y + offsetY < height_ && x + offsetX >= 0 &&
              x + offsetX < width_ &&
              image[width_ * (y + offsetY) + x + offsetX] >= centerValue)
            censusCode += 1;
          // Census区域中>=中心点值时，就将censusCode加一，块内点的相对关系不同时censusCode值不同。相当于一种编码手段。
        }
      }
      censusImage[width_ * y + x] =
          censusCode; //将每块区域的统计值填入到相应的位置之中。
    }
  }
}

void SGMFlow::calcTopRowCost(
    unsigned char *&leftSobelRow, int *&leftCensusRow,
    unsigned char *&leftplusSobelRow, int *&leftplusCensusRow,
    unsigned short *costImageRow, unsigned char *&leftSobelImage,
    unsigned char *&leftplusSobelImage, const int *leftCensusImage,
    const int *leftplusCensusImage, const bool calcLeft) {
  //这个函数标注的计算顶行的cost，顶行因为在边缘，所以要特殊处理。
  //因为最初给的时候这个位置的rowIndex=0，所以直接让leftSobelImage的指针和row一样就行了。
  for (int rowIndex = 0; rowIndex <= aggregationWindowRadius_; ++rowIndex) {
    //都是在计算不超过窗的半径的cost。
    int rowAggregatedCostIndex =
        std::min(rowIndex, height_ - 1) % (aggregationWindowRadius_ * 2 + 2);
    //行数和最大行数两者之前取个最小，这个是防止程序出错做的防护。
    unsigned short *rowAggregatedCostCurrent =
        rowAggregatedCost_ + rowAggregatedCostIndex * width_ * vzratioTotal_;
    //这个rowAggregatedCost_是哪里来的?算了，计算SAD的时候也没有用到那就先搁置不�����了。

    // calcPixelwiseSAD(leftSobelRow, leftplusSobelRow, 0);
    calcPixelwiseSADHamming(leftSobelRow, leftplusSobelRow, leftSobelImage,
                            leftplusSobelImage, leftCensusRow,
                            leftplusCensusRow, leftCensusImage,
                            leftplusCensusImage, rowIndex);
    // addPixelwiseHamming(leftCensusRow, leftplusCensusRow);
    memset(rowAggregatedCostCurrent, 0, vzratioTotal_ * sizeof(unsigned short));
    // x = 0
    for (int x = 0; x <= aggregationWindowRadius_; ++x) {
      int scale = x == 0 ? aggregationWindowRadius_ + 1 : 1;
      for (int wp = 0; wp < vzratioTotal_; ++wp) {
        rowAggregatedCostCurrent[wp] += static_cast<unsigned short>(
            pixelwiseCostRow_[vzratioTotal_ * x + wp] * scale);
        //此处本身aggregationWindowRadius_=2，这里如果是起始点，因为其左边没有别的点了
        //所以就把自己乘了三遍，而其他的位置因为左右都有点，所以聚合的时候是三个点的cost加和。
      }
    }
    // x = 1...width-1
    for (int x = 1; x < width_; ++x) {
      // cost aggregation 时框内右边的点是addPixelwiseCost，
      //左边的点是subPixelwiseCost，下面这两句都是考虑了右边界和左边界的情况
      const unsigned char *addPixelwiseCost =
          pixelwiseCostRow_ +
          std::min((x + aggregationWindowRadius_) * vzratioTotal_,
                   (width_ - 1) * vzratioTotal_);
      const unsigned char *subPixelwiseCost =
          pixelwiseCostRow_ +
          std::max((x - aggregationWindowRadius_ - 1) * vzratioTotal_, 0);
      for (int wp = 0; wp < vzratioTotal_; ++wp) {
        rowAggregatedCostCurrent[vzratioTotal_ * x + wp] =
            static_cast<unsigned short>(
                rowAggregatedCostCurrent[vzratioTotal_ * (x - 1) + wp] +
                addPixelwiseCost[wp] - subPixelwiseCost[wp]);
      }
    }
    // Add to cost
    int scale = rowIndex == 0 ? aggregationWindowRadius_ + 1 : 1;
    for (int i = 0; i < width_ * vzratioTotal_; ++i) {
      costImageRow[i] += rowAggregatedCostCurrent[i] * scale;
    }
    leftSobelRow += widthStep_;
    leftplusSobelRow += widthStep_;
    leftCensusRow += width_;
    leftplusCensusRow += width_;
  }
}

void SGMFlow::calcRowCosts(
    unsigned char *&leftSobelRow, int *&leftCensusRow,
    unsigned char *&leftplusSobelRow, int *&leftplusCensusRow,
    unsigned short *costImageRow, unsigned char *&leftSobelImage,
    unsigned char *&leftplusSobelImage, const int *leftCensusImage,
    const int *leftplusCensusImage, const bool calcLeft) {
  const int widthStepCost = width_ * vzratioTotal_;
  const __m128i registerZero = _mm_setzero_si128();

  for (int y = 1; y < height_; ++y) {
    int addRowIndex = y + aggregationWindowRadius_;
    int addRowAggregatedCostIndex =
        std::min(addRowIndex, height_ - 1) % (aggregationWindowRadius_ * 2 + 2);
    unsigned short *addRowAggregatedCost =
        rowAggregatedCost_ + width_ * vzratioTotal_ * addRowAggregatedCostIndex;
    //移动指针指向下一行的起始位置
    if (addRowIndex < height_) {
      // calcPixelwiseSAD(leftSobelRow, leftplusSobelRow, y);
      calcPixelwiseSADHamming(leftSobelRow, leftplusSobelRow, leftSobelImage,
                              leftplusSobelImage, leftCensusRow,
                              leftplusCensusRow, leftCensusImage,
                              leftplusCensusImage, y);
      // std::cout << "enter the Row SAD, at y=" << y << std::endl;
      // addPixelwiseHamming(leftCensusRow, leftplusCensusRow);
      memset(addRowAggregatedCost, 0, vzratioTotal_ * sizeof(unsigned short));
      // x = 0
      for (int x = 0; x <= aggregationWindowRadius_; ++x) {
        int scale = x == 0 ? aggregationWindowRadius_ + 1 : 1;
        for (int wp = 0; wp < vzratioTotal_; ++wp) {
          addRowAggregatedCost[wp] += static_cast<unsigned short>(
              pixelwiseCostRow_[vzratioTotal_ * x + wp] * scale);
        }
      }
      // x = 1...witdth-1
      int subRowAggregatedCostIndex =
          std::max(y - aggregationWindowRadius_ - 1, 0) %
          (aggregationWindowRadius_ * 2 + 2);
      const unsigned short *subRowAggregatedCost =
          rowAggregatedCost_ +
          width_ * vzratioTotal_ * subRowAggregatedCostIndex;
      const unsigned short *previousCostRow = costImageRow - widthStepCost;
      for (int x = 1; x < width_; ++x) {
        const unsigned char *addPixelwiseCost =
            pixelwiseCostRow_ +
            std::min((x + aggregationWindowRadius_) * vzratioTotal_,
                     (width_ - 1) * vzratioTotal_);
        const unsigned char *subPixelwiseCost =
            pixelwiseCostRow_ +
            std::max((x - aggregationWindowRadius_ - 1) * vzratioTotal_, 0);

        for (int wp = 0; wp < vzratioTotal_; wp += 16) {
          // wp = 2;
          // std::cout << "point -> value:" << int(*(addPixelwiseCost + wp))
          //           << std::endl;
          __m128i registerAddPixelwiseLow = _mm_load_si128(
              reinterpret_cast<const __m128i *>(addPixelwiseCost + wp));
          __m128i registerAddPixelwiseHigh =
              _mm_unpackhi_epi8(registerAddPixelwiseLow, registerZero);
          registerAddPixelwiseLow =
              _mm_unpacklo_epi8(registerAddPixelwiseLow, registerZero);
          __m128i registerSubPixelwiseLow = _mm_load_si128(
              reinterpret_cast<const __m128i *>(subPixelwiseCost + wp));
          __m128i registerSubPixelwiseHigh =
              _mm_unpackhi_epi8(registerSubPixelwiseLow, registerZero);
          registerSubPixelwiseLow =
              _mm_unpacklo_epi8(registerSubPixelwiseLow, registerZero);
          // Low
          __m128i registerAddAggregated =
              _mm_load_si128(reinterpret_cast<const __m128i *>(
                  addRowAggregatedCost + vzratioTotal_ * (x - 1) + wp));
          registerAddAggregated = _mm_adds_epi16(
              _mm_subs_epi16(registerAddAggregated, registerSubPixelwiseLow),
              registerAddPixelwiseLow);
          __m128i registerCost =
              _mm_load_si128(reinterpret_cast<const __m128i *>(
                  previousCostRow + vzratioTotal_ * x + wp));
          registerCost = _mm_adds_epi16(
              _mm_subs_epi16(
                  registerCost,
                  _mm_load_si128(reinterpret_cast<const __m128i *>(
                      subRowAggregatedCost + vzratioTotal_ * x + wp))),
              registerAddAggregated);
          _mm_store_si128(reinterpret_cast<__m128i *>(addRowAggregatedCost +
                                                      vzratioTotal_ * x + wp),
                          registerAddAggregated);
          _mm_store_si128(reinterpret_cast<__m128i *>(costImageRow +
                                                      vzratioTotal_ * x + wp),
                          registerCost);

          // High
          registerAddAggregated =
              _mm_load_si128(reinterpret_cast<const __m128i *>(
                  addRowAggregatedCost + vzratioTotal_ * (x - 1) + wp + 8));
          registerAddAggregated = _mm_adds_epi16(
              _mm_subs_epi16(registerAddAggregated, registerSubPixelwiseHigh),
              registerAddPixelwiseHigh);
          registerCost = _mm_load_si128(reinterpret_cast<const __m128i *>(
              previousCostRow + vzratioTotal_ * x + wp + 8));
          registerCost = _mm_adds_epi16(
              _mm_subs_epi16(
                  registerCost,
                  _mm_load_si128(reinterpret_cast<const __m128i *>(
                      subRowAggregatedCost + vzratioTotal_ * x + wp + 8))),
              registerAddAggregated);
          _mm_store_si128(reinterpret_cast<__m128i *>(addRowAggregatedCost +
                                                      vzratioTotal_ * x + wp +
                                                      8),
                          registerAddAggregated);
          _mm_store_si128(reinterpret_cast<__m128i *>(
                              costImageRow + vzratioTotal_ * x + wp + 8),
                          registerCost);
        }
      }
    }
    leftSobelRow += widthStep_;
    leftplusSobelRow += widthStep_;
    leftCensusRow += width_;
    leftplusCensusRow += width_;
    costImageRow += widthStepCost;
  }
}

void SGMFlow::calcPixelwiseSADHamming(
    const unsigned char *leftSobelRow, const unsigned char *leftplusSobelRow,
    const unsigned char *leftSobelImage,
    const unsigned char *leftplusSobelImage, const int *leftCensusRow,
    const int *leftplusCensusRow, const int *leftCensusImage,
    const int *leftplusCensusImage, const int yIndex, const bool calcLeft) {
  //这个函数作为一个通用版本，既可以计算leftCostImage，又可以用于一致性校验计算leftplus图片
  float vMax = 0.45;

  if (calcLeft == true) {
    //这里面之所以与原来不同，计算leftSobelRow而不是leftPlus的，因为flow寻找matching点
    //的时候是沿着极线方向进行搜索的，所以没有办法很好的预先计算出leftplus上被匹配的点。
    //所以在这里我取消所有预计算，所有都是现取现算。
    int y = yIndex;
    for (int x = 0; x < width_; ++x) {
      int leftCenterValue = leftSobelRow[x];
      double uwTransX, uwTransY;
      // calculate the rotation
      uwTransX = (f * wy_t - wz_t * y + wy_t * x * x / f - wx_t * x * y / f);
      uwTransY = (-f * wx_t + wz_t * x + wy_t * x * y / f - wx_t * y * y / f);
      // double a1, a2, a3, a4, a5;
      // a1 = 1.796551571041166;
      // a2 = -1.376330362339697;
      // a3 = -0.001713217331441;
      // a4 = -0.000071754795906;
      // a5 = 0.000143306169912;
      // uwTransX = a1 - a3 * y + a4 * x * x + a5 * x * y;
      // uwTransY = a2 + a3 * x + a4 * x * y + a5 * y * y;
      // distance between p and epipole o;
      double distancePEpi = sqrt(pow(x + uwTransX - epipoleX, 2) +
                                 pow(y + uwTransY - epipoleY, 2));
      double directionX = x - epipoleX;
      double directionY = y - epipoleY;
      double length = sqrt(pow(directionX, 2) + pow(directionY, 2));
      directionX = directionX / length;
      directionY = directionY / length;

      int xf_ = x + std::round(directionX) < width_ - 1 &&
                        x + std::round(directionX) > 0
                    ? x + std::round(directionX)
                    : x;
      int yf_ = y + std::round(directionY) < height_ - 1 &&
                        y + std::round(directionY) > 0
                    ? y + std::round(directionY)
                    : y;
      int xb_ = x - std::round(directionX) < width_ - 1 &&
                        x - std::round(directionX) > 0
                    ? x - std::round(directionX)
                    : x;
      int yb_ = y - std::round(directionY) < height_ - 1 &&
                        y - std::round(directionY) > 0
                    ? y - std::round(directionY)
                    : y;

      int leftHalfLeftValue =
          x > 0 ? (leftCenterValue + leftSobelImage[widthStep_ * yf_ + xf_]) / 2
                : leftCenterValue;

      int leftHalfRightValue =
          x < width_ - 1
              ? (leftCenterValue + leftSobelImage[widthStep_ * yb_ + xb_]) / 2
              : leftCenterValue;
      int leftMinValue = std::min(leftHalfLeftValue, leftHalfRightValue);
      leftMinValue = std::min(leftMinValue, leftCenterValue);
      int leftMaxValue = std::max(leftHalfLeftValue, leftHalfRightValue);
      leftMaxValue = std::max(leftMaxValue, leftCenterValue);

      for (int wp = 0; wp < vzratioTotal_; ++wp) {
        double distanceRRhat = fabs(distancePEpi * wp * (vMax / vzratioTotal_) /
                                    (1 - wp * (vMax / vzratioTotal_)));
        int xPlus = x + std::round( uwTransX ) + std::ceil(directionX * distanceRRhat);
        int yPlus = y + std::round( uwTransY ) + std::ceil(directionY * distanceRRhat);
        int wpf = wp > 0 ? wp - 1 : 0;
        int wpb = wp < vzratioTotal_ - 1 ? wp + 1 : wp;
        double distf = fabs(distancePEpi * wpf * (vMax / vzratioTotal_) /
                            (1 - wpf * (vMax / vzratioTotal_)));
        double distb = fabs(distancePEpi * wpb * (vMax / vzratioTotal_) /
                            (1 - wpb * (vMax / vzratioTotal_)));
        int xf = x + std::ceil(uwTransX + directionX * distf);
        int yf = y + std::ceil(uwTransY + directionY * distf);
        int xb = x + std::ceil(uwTransX + directionX * distb);
        int yb = y + std::ceil(uwTransY + directionY * distb);
        xf = xf > 0 && xf < width_ - 1 ? xf : xPlus;
        yf = yf > 0 && yf < height_ - 1 ? yf : yPlus;
        xb = xb > 0 && xb < width_ - 1 ? xb : xPlus;
        yb = yb > 0 && yb < height_ - 1 ? yb : yPlus;

//        int xf = xf_ + std::round(uwTransX)+std::ceil(directionX * distanceRRhat);
//        int yf = yf_ + std::round(uwTransY)+std::ceil(directionY * distanceRRhat);
//        int xb = xb_ + std::round(uwTransX)+std::ceil(directionX * distanceRRhat);
//        int yb = yb_ + std::round(uwTransY)+std::ceil(directionY * distanceRRhat);
//        xf = xf > 0 && xf < width_ - 1 ? xf : xPlus;
//        yf = yf > 0 && yf < height_ - 1 ? yf : yPlus;
//        xb = xb > 0 && xb < width_ - 1 ? xb : xPlus;
//        yb = yb > 0 && yb < height_ - 1 ? yb : yPlus;

          if (xPlus >= 0 && yPlus >= 0 && xPlus < width_ && yPlus < height_) {
          int leftplusCenterValue =
              leftplusSobelImage[widthStep_ * yPlus + width_ - 1 - xPlus];
          int leftplusHalfLeftValue =
              xPlus > 0
                  ? (leftplusCenterValue +
                     leftplusSobelImage[widthStep_ * yf + width_ - 1 - xf]) /
                        2
                  : leftplusCenterValue;
          int leftplusHalfRightValue =
              xPlus < width_ - 1
                  ? (leftplusCenterValue +
                     leftplusSobelImage[widthStep_ * yb + width_ - 1 - xb]) /
                        2
                  : leftplusCenterValue;
          int leftplusMinValue =
              std::min(leftplusHalfLeftValue, leftplusHalfRightValue);
          leftplusMinValue = std::min(leftplusMinValue, leftplusCenterValue);
          int leftplusMaxValue =
              std::max(leftplusHalfLeftValue, leftplusHalfRightValue);
          leftplusMaxValue = std::max(leftplusMaxValue, leftplusCenterValue);

          int costLtoR = std::max(0, leftCenterValue - leftplusMaxValue);
          costLtoR = std::max(costLtoR, leftplusMinValue - leftCenterValue);
          int costRtoL = std::max(0, leftplusCenterValue - leftMaxValue);
          costRtoL = std::max(costRtoL, leftMinValue - leftplusCenterValue);
          int costValue = std::min(costLtoR, costRtoL);
          pixelwiseCostRow_[vzratioTotal_ * x + wp] = costValue;
          addPixelwiseHamming(leftCensusRow, leftplusCensusRow, leftCensusImage,
                              leftplusCensusImage, true, x, xPlus, yPlus, wp);
              int a =pixelwiseCostRow_[vzratioTotal_ * x + wp];
              int b;
        } else {
          if (wp == 0) {
            pixelwiseCostRow_[vzratioTotal_ * xPlus] = 15;
            wp++;
          }
          for (int wpRemains = wp; wpRemains < vzratioTotal_ - 1; ++wpRemains) {
            pixelwiseCostRow_[vzratioTotal_ * xPlus + wpRemains] =
                pixelwiseCostRow_[vzratioTotal_ * xPlus + wpRemains - 1];
          }
          pixelwiseCostRow_[vzratioTotal_ * xPlus + vzratioTotal_ - 1] = 85;
          break;
        }
      }
    }
  } // left cost calc end
  else {
    int yPlus = yIndex;
    for (int xPlus = 0; xPlus < width_; ++xPlus) {
      //这个for是�����leftPlus开始计算���������
      int leftplusCenterValue =
          leftplusSobelImage[widthStep_ * yPlus + width_ - xPlus - 1];
      int leftHalfLeftValue =
          xPlus > 0
              ? (leftplusCenterValue +
                 leftplusSobelImage[widthStep_ * yPlus + width_ - (xPlus - 1) -
                                    1]) /
                    2
              : leftplusCenterValue;
      int leftHalfRightValue =
          xPlus < width_ - 1
              ? (leftplusCenterValue +
                 leftplusSobelImage[widthStep_ * yPlus + width_ - (xPlus + 1) -
                                    1]) /
                    2
              : leftplusCenterValue;
      int leftplusMinValue = std::min(leftHalfLeftValue, leftHalfRightValue);
      leftplusMinValue = std::min(leftplusMinValue, leftplusCenterValue);
      int leftplusMaxValue = std::max(leftHalfLeftValue, leftHalfRightValue);
      leftplusMaxValue = std::max(leftplusMaxValue, leftplusCenterValue);
      for (int wp = 0; wp <= vzratioTotal_; ++wp) {
        double uwTransX, uwTransY;
        uwTransX = (f * wy_inv - wz_inv * yPlus + wy_inv * xPlus * xPlus / f -
                    wx_inv * xPlus * yPlus / f);
        uwTransY = (-f * wx_inv + wz_inv * xPlus + wy_inv * xPlus * yPlus / f -
                    wx_inv * yPlus * yPlus / f);
        double distancePEpi = sqrt(pow(xPlus + uwTransX - epipoleX, 2) +
                                   pow(yPlus + uwTransY - epipoleY, 2));
        // double distanceRRhat = distancePEpi * wp * (vMax / vzratioTotal_) /(1
        // - wp * (vMax / vzratioTotal_));
        double distanceRRhat = fabs(distancePEpi * wp * (vMax / vzratioTotal_) /
                                    (1 - wp * (vMax / vzratioTotal_)));
        //这个是计算的时候应该是朝着极点走的。
        double directionX = epipoleX - xPlus;
        double directionY = epipoleY - yPlus;
        double length = sqrt(pow(directionX, 2) + pow(directionY, 2));
        directionX = directionX / length;
        directionY = directionY / length;
        int x = xPlus + uwTransX + directionX * distanceRRhat;
        int y = xPlus + uwTransY + directionY * distanceRRhat;

        if (x >= 0 && y >= 0 && x < width_ && y < height_) {
          int leftCenterValue = leftSobelImage[widthStep_ * y + x];
          int leftHalfLeftValue =
              x > 0 ? (leftCenterValue + leftSobelRow[x - 1]) / 2
                    : leftCenterValue;
          int leftHalfRightValue =
              x < width_ - 1 ? (leftCenterValue + leftSobelRow[x + 1]) / 2
                             : leftCenterValue;
          int leftMinValue = std::min(leftHalfLeftValue, leftHalfRightValue);
          leftMinValue = std::min(leftMinValue, leftCenterValue);
          int leftMaxValue = std::max(leftHalfLeftValue, leftHalfRightValue);
          leftMaxValue = std::max(leftMaxValue, leftCenterValue);

          int costLtoR = std::max(0, leftCenterValue - leftplusMaxValue);
          costLtoR = std::max(costLtoR, leftplusMinValue - leftCenterValue);
          int costRtoL = std::max(0, leftplusCenterValue - leftMaxValue);
          costRtoL = std::max(costRtoL, leftMinValue - leftplusCenterValue);
          int costValue = std::min(costLtoR, costRtoL);
          pixelwiseCostRow_[vzratioTotal_ * xPlus + wp] = costValue;
          addPixelwiseHamming(leftCensusRow, leftplusCensusRow, leftCensusImage,
                              leftplusCensusImage, false, xPlus, x, y, wp);

        } else {
          if (wp == 0) {
            pixelwiseCostRow_[vzratioTotal_ * xPlus] = 255;
            pixelwiseCostRow_[vzratioTotal_ * xPlus + 1] = 0;
            for (int wpRemains = 2; wpRemains < vzratioTotal_; ++wpRemains) {
              pixelwiseCostRow_[vzratioTotal_ * xPlus + wpRemains] =
                  pixelwiseCostRow_[vzratioTotal_ * xPlus + wpRemains - 1];
            }
          }
          for (int wpRemains = wp; wpRemains < vzratioTotal_; ++wpRemains) {
            pixelwiseCostRow_[vzratioTotal_ * xPlus + wpRemains] =
                pixelwiseCostRow_[vzratioTotal_ * xPlus + wpRemains - 1];
          }
          break;
        }
      }
    } // for xPlus < width_
  }   // else
}

void SGMFlow::addPixelwiseHamming(const int *leftCensusRow,
                                  const int *leftplusCensusRow,
                                  const int *leftCensusImage,
                                  const int *leftplusCensusImage,
                                  const bool calcLeft, const int xBase,
                                  const int xMatching, const int yMatching,
                                  const int wp) {

  int leftCensusCode, leftplusCensusCode;
  int hammingDistance = 0;
  if (calcLeft == true) {
    leftCensusCode = leftCensusRow[xBase];
    leftplusCensusCode = leftplusCensusImage[yMatching * width_ + xMatching];
  } else {
    leftplusCensusCode = leftplusCensusRow[xBase];
    leftCensusCode = leftCensusImage[yMatching * width_ + xMatching];
  }
  hammingDistance = static_cast<int>(__builtin_popcount(
      static_cast<unsigned int>(leftCensusCode ^ leftplusCensusCode)));
  pixelwiseCostRow_[vzratioTotal_ * xBase + wp] +=
      static_cast<unsigned char>(hammingDistance * censusWeightFactor_);
  //这里不再写hamming distance
  // cost逐个像素填充的原因是�����������������������如���找不到对应点，那也没有参考���
  // distance
  // 能够填充���如果有的话，计算sad的时候是逐个点计算完之后就计算hamming
  //所以和前面的的一样的话，也相当于填充了
}

void SGMFlow::computeLeftPlusCostImage() {
  // const int widthStepCost = width_ * vzratioTotal_;
  //这个几个关键的内存块等到leftplus的cost计算完之后再将其释放。
  unsigned char *leftSobelRow = leftSobelImage;
  unsigned char *leftplusSobelRow = leftplusSobelImage;
  int *leftCensusRow = leftCensusImage;
  int *leftplusCensusRow = leftplusCensusImage;
  unsigned short *costImageRow = leftplusCostImage_;
  calcTopRowCost(leftSobelRow, leftCensusRow, leftplusSobelRow,
                 leftplusCensusRow, costImageRow, leftSobelImage,
                 leftplusSobelImage, leftCensusImage, leftplusCensusImage,
                 false);
  costImageRow += width_ * vzratioTotal_;
  calcRowCosts(leftSobelRow, leftCensusRow, leftplusSobelRow, leftplusCensusRow,
               costImageRow, leftSobelImage, leftplusSobelImage,
               leftCensusImage, leftplusCensusImage, false);

  _mm_free(leftSobelImage);
  _mm_free(leftplusSobelImage);
  free(leftCensusImage);
  free(leftplusCensusImage);
}

void SGMFlow::performSGM(unsigned short *costImage,
                         unsigned short *vzratioImage) {
  const short costMax = SHRT_MAX;
  //因为SGM那篇文章里面介绍了Aggregation cost是有最大值的。
  int widthStepCostImage = width_ * vzratioTotal_;
  //上面这个变量里面记录的应该是，这一行的偏移量，也就是步长
  short *costSums =
      sgmBuffer_; // costsum也是指向了一块内存，这块内存很大，基本上能存sgm所需要的所有的大小
  memset(costSums, 0, costSumBufferSize_ * sizeof(short));

  short **pathCosts = new short *[pathRowBufferTotal_];
  short **pathMinCosts = new short *[pathRowBufferTotal_];

  const int processPassTotal = 2;
  for (int processPassCount = 0; processPassCount < processPassTotal;
       ++processPassCount) {
    int startX, endX, stepX;
    int startY, endY, stepY;
    if (processPassCount == 0) {
      startX = 0;
      endX = width_;
      stepX = 1;
      startY = 0;
      endY = height_;
      stepY = 1;
    } else {
      startX = width_ - 1;
      endX = -1;
      stepX = -1;
      startY = height_ - 1;
      endY = -1;
      stepY = -1;
    }
    // pathRowBufferTotal_ = 2;
    // 下面这个for都是在初始化，没有做别的事情
    for (int i = 0; i < pathRowBufferTotal_; ++i) {
      pathCosts[i] = costSums + costSumBufferSize_ + pathCostBufferSize_ * i +
                     pathVZratioSize_ + 8;
      //这是在提供位置，并非是计算赋值操作.costSums是一块大内存的起始位置
      //这个位置的pathCostBufferSize_一偏移一行就计算完了。

      memset(pathCosts[i] - pathVZratioSize_ - 8, 0,
             pathCostBufferSize_ * sizeof(short));
      //这里之所以减去是为了应对下面的负���偏���
      //所以���里剪完之后，从pathCosts[i] = costSums +
      // costSumBufferSize_开���，������块内存全是0
      pathMinCosts[i] = costSums + costSumBufferSize_ +
                        pathCostBufferSize_ * pathRowBufferTotal_ +
                        pathMinCostBufferSize_ * i + pathTotal_ * 2;
      //可以看到pathMinCosts用的是pathCosts之后的一块空间。
      // pathCostBufferSize_ *
      // pathRowBufferTotal_相当于是把pathCosts的空间让出来了
      //一个pathMinCosts用的空间是pathMinCostBufferSize_这么大
      //也就是一行的path那么大
      memset(pathMinCosts[i] - pathTotal_, 0,
             pathMinCostBufferSize_ * sizeof(short));
      //可以看到他一开始指的位置靠后了16，但是初始化的时候只初始化到靠前
      // 8个位置，可能因为pathCosts也靠后了8个位置，他在这里把位置让出来了
    }

    for (int y = startY; y != endY; y += stepY) {
      unsigned short *pixelCostRow = costImage + widthStepCostImage * y;
      short *costSumRow = costSums + costSumBufferRowSize_ * y;
      // 难道最后空出来的那部分8个位置，存储的是每一行最小cost的加和
      // 下面这一块也都是在初始化，没必要再看了
      memset(pathCosts[0] - pathVZratioSize_ - 8, 0,
             pathVZratioSize_ * sizeof(short));
      memset(pathCosts[0] + width_ * pathVZratioSize_ - 8, 0,
             pathVZratioSize_ * sizeof(short));
      memset(pathMinCosts[0] - pathTotal_, 0, pathTotal_ * sizeof(short));
      memset(pathMinCosts[0] + width_ * pathTotal_, 0,
             pathTotal_ * sizeof(short));
      //这里的初始化也非常的简单，就是按位置来进行初始化，都初始化为0，两两为�����组，
      //一组里面的值都是对应的两个部分，从�����算的空间，和从���下计算���空间�����
      for (int x = startX; x != endX; x += stepX) {
        int pathMinX =
            x * pathTotal_; //这里表示的是点的位置x8条path位置����������移量）
        int pathX =
            pathMinX * vzratioSize_; //点的位置x8条pathx所有的视差等级（偏移量）
        int previousPathMin0 = pathMinCosts[0][pathMinX - stepX * pathTotal_] +
                               smoothnessPenaltyLarge_;
        //上面初始化的时候-pathTotal_所以把stepX *
        // pathTotal_�������个位置������出来了
        //这句话的意思是，将之前的最小的path的最小的cost拿出来加上一个大的平滑惩罚，
        //为什么默认在���个位置是视差等级查1个以上的的位置
        int previousPathMin2 =
            pathMinCosts[1][pathMinX + 2] + smoothnessPenaltyLarge_;
        //���意这两个变量不是指针了，已经是整形的数值了
        //这手看着贼迷乱
        short *previousPathCosts0 =
            pathCosts[0] + pathX - stepX * pathVZratioSize_;
        short *previousPathCosts2 = pathCosts[1] + pathX + vzratioSize_ * 2;
        //下面���一���是���，���一���起���点���位���，�����������放���最���的cost
        previousPathCosts0[-1] = previousPathCosts0[vzratioTotal_] = costMax;
        previousPathCosts2[-1] = previousPathCosts2[vzratioTotal_] = costMax;

        short *pathCostCurrent = pathCosts[0] + pathX;
        const unsigned short *pixelCostCurrent =
            pixelCostRow + vzratioTotal_ * x;
        short *costSumCurrent = costSumRow + vzratioTotal_ * x;
        // pixelCostRow 指向的是CostImage的起始位���
        __m128i regPenaltySmall =
            _mm_set1_epi16(static_cast<short>(smoothnessPenaltySmall_));

        __m128i regPathMin0, regPathMin2;
        regPathMin0 = _mm_set1_epi16(static_cast<short>(previousPathMin0));
        regPathMin2 = _mm_set1_epi16(static_cast<short>(previousPathMin2));
        __m128i regNewPathMin = _mm_set1_epi16(costMax);

        for (int d = 0; d < vzratioTotal_; d += 8) {
          __m128i regPixelCost = _mm_load_si128(
              reinterpret_cast<const __m128i *>(pixelCostCurrent + d));
          //因为pixelCostCurrent也是一个指针，所以给了偏移量d之后取值为，在视差为d时cost。
          __m128i regPathCost0, regPathCost2;
          regPathCost0 = _mm_load_si128(
              reinterpret_cast<const __m128i *>(previousPathCosts0 + d));
          regPathCost2 = _mm_load_si128(
              reinterpret_cast<const __m128i *>(previousPathCosts2 + d));
          //因为视差值相差1，加小惩罚，这个相差一包括大一和小一，所以此处有d+1和d-1
          //并且比较了，直接视差取d和相差1加small惩罚项到底cost更小
          regPathCost0 = _mm_min_epi16(
              regPathCost0,
              _mm_adds_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i *>(
                                 previousPathCosts0 + d - 1)),
                             regPenaltySmall));
          //这里是先将之前�������������path
          regPathCost0 = _mm_min_epi16(
              regPathCost0,
              _mm_adds_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i *>(
                                 previousPathCosts0 + d + 1)),
                             regPenaltySmall));
          regPathCost2 = _mm_min_epi16(
              regPathCost2,
              _mm_adds_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i *>(
                                 previousPathCosts2 + d - 1)),
                             regPenaltySmall));
          regPathCost2 = _mm_min_epi16(
              regPathCost2,
              _mm_adds_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i *>(
                                 previousPathCosts2 + d + 1)),
                             regPenaltySmall));
          // regPathMin0为之前path最小的cost，���������在
          regPathCost0 = _mm_min_epi16(regPathCost0, regPathMin0);
          //因为path本身是不重要的���而在每一步取的最小的值是最重要的，所以而且他在遍历这个视差的时候
          //相当于是站在一个点处，去寻找这个点处，对应最小cost的视差���������级。
          //因为这个还是逐行处理的
          regPathCost0 = _mm_adds_epi16(
              _mm_subs_epi16(regPathCost0, regPathMin0), regPixelCost);
          //这一步应该就是作者在文章里面提到的，为了保持这个cost不要持���增大，所以要�������次减去
          //之前最小路径的cost，这样在每一个位置处才有最小值和最大值。
          regPathCost2 = _mm_min_epi16(regPathCost2, regPathMin2);
          regPathCost2 = _mm_adds_epi16(
              _mm_subs_epi16(regPathCost2, regPathMin2), regPixelCost);

          _mm_store_si128(reinterpret_cast<__m128i *>(pathCostCurrent + d),
                          regPathCost0);
          _mm_store_si128(reinterpret_cast<__m128i *>(pathCostCurrent + d +
                                                      vzratioSize_ * 2),
                          regPathCost2);
          //上面这两������将计算的path的cost，���储���相应的内存位���������������
          __m128i regMin02 =
              _mm_min_epi16(_mm_unpacklo_epi16(regPathCost0, regPathCost2),
                            _mm_unpackhi_epi16(regPathCost0, regPathCost2));
          // extern __m128i _mm_unpacklo_epi16(__m128i _A, __m128i _B);
          //返回一个__m128i������������������������������������������存器，它将寄存器_A和寄存���_B的低64bit数以32bit为单位交织在一块。
          //例如，_A=(_A0,_A1,_A2,_A3,_A4,_A5,_A6,_A7),_B=(_B0,_B1,_B2,_B3,_B4,_B5,_B6,_B7),
          //其中_Ai,_Bi为16bit整数，_A0,_B0为低位，返回结果为(_A0,_A1,_B0,_B1,_A2,_A3,_B2,_B3),
          // r0=_A0, r1=_B0, r2=_A1, r3=_B1
          // extern __m128i _mm_unpackhi_epi16(__m128i _A, __m128i _B);
          //返回一个__m128i的寄存器，它将寄存器_A和寄存器_B的高64bit数以32bit为单位交织在一块。
          //例如，_A=(_A0,_A1,_A2,_A3,_A4,_A5,_A6,_A7),_B=(_B0,_B1,_B2,_B3,_B4,_B5,_B6,_B7),
          //其中_Ai,_Bi为16bit整数，_A0,_B0为低位，返回结果���(_A4,_A5,_B4,_B5,_A6,_A7,_B6,_B7),
          // r0=_A2, r1=_B2, r2=_A3, r3=_B3
          //
          // 我个人认���这个0和2
          // �������做的事情是一模一样的，他在计��������时������一遍这���东西讲道理应该是没什么用的
          // 因���他的初始值都是一样的，所以计算出来的东西也应该都是一样的，看她把值存储在了从右下角位置开始
          // 的内存位置，应该是说他第二遍从右下脚开始计算的时候没准会用到2的结���。
          regMin02 = _mm_min_epi16(_mm_unpacklo_epi16(regMin02, regMin02),
                                   _mm_unpackhi_epi16(regMin02, regMin02));
          regNewPathMin = _mm_min_epi16(regNewPathMin, regMin02);

          __m128i regCostSum = _mm_load_si128(
              reinterpret_cast<const __m128i *>(costSumCurrent + d));

          regCostSum = _mm_adds_epi16(regCostSum, regPathCost0);
          regCostSum = _mm_adds_epi16(regCostSum, regPathCost2);

          _mm_store_si128(reinterpret_cast<__m128i *>(costSumCurrent + d),
                          regCostSum);
        }

        regNewPathMin =
            _mm_min_epi16(regNewPathMin, _mm_srli_si128(regNewPathMin, 8));
        _mm_storel_epi64(
            reinterpret_cast<__m128i *>(&pathMinCosts[0][pathMinX]),
            regNewPathMin);
        // extern __m128i _mm_srli_si128(__m128i _A, int _Imm);
        //返回一个__m128i的寄存器，将���存���_A中���8个16bit整���按照_Count���行相�����的逻辑右���，
        //移�������填�����值���������0,r0=srl(_A0, _Count), r1=srl(_A1, _Count),
        //...
        // r7=srl(_A7, _Count),
        // shifting in zeros
      }

      if (processPassCount == processPassTotal - 1) {
        unsigned short *disparityRow = vzratioImage + width_ * y;

        for (int x = 0; x < width_; ++x) {
          short *costSumCurrent = costSumRow + vzratioTotal_ * x;
          int bestSumCost = costSumCurrent[0];
          int bestDisparity = 0;
          for (int d = 1; d < vzratioTotal_; ++d) {
            if (costSumCurrent[d] < bestSumCost) {
              bestSumCost = costSumCurrent[d];
              bestDisparity = d;
            }
          }

          if (bestDisparity > 0 && bestDisparity < vzratioTotal_ - 1) {
            int centerCostValue = costSumCurrent[bestDisparity];
            int leftCostValue = costSumCurrent[bestDisparity - 1];
            int rightCostValue = costSumCurrent[bestDisparity + 1];
            if (rightCostValue < leftCostValue) {
              bestDisparity = static_cast<int>(
                  bestDisparity * vzratioFactor_ +
                  static_cast<double>(rightCostValue - leftCostValue) /
                      (centerCostValue - leftCostValue) / 2.0 * vzratioFactor_ +
                  0.5);
            } else {
              bestDisparity = static_cast<int>(
                  bestDisparity * vzratioFactor_ +
                  static_cast<double>(rightCostValue - leftCostValue) /
                      (centerCostValue - rightCostValue) / 2.0 *
                      vzratioFactor_ +
                  0.5);
            }
          } else {
            bestDisparity = static_cast<int>(bestDisparity * vzratioFactor_);
          }

          disparityRow[x] = static_cast<unsigned short>(bestDisparity);
        }
      }

      std::swap(pathCosts[0], pathCosts[1]);
      std::swap(pathMinCosts[0], pathMinCosts[1]);
    }
  }
  delete[] pathCosts;
  delete[] pathMinCosts;

  speckleFilter(100, static_cast<int>(2 * vzratioFactor_), vzratioImage);
}

void SGMFlow::speckleFilter(const int maxSpeckleSize, const int maxDifference,
                            unsigned short *image) const {
  std::vector<int> labels(width_ * height_, 0);
  std::vector<bool> regionTypes(1);
  regionTypes[0] = false;

  int currentLabelIndex = 0;

  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      int pixelIndex = width_ * y + x;
      if (image[width_ * y + x] != 0) {
        if (labels[pixelIndex] > 0) {
          if (regionTypes[labels[pixelIndex]]) {
            image[width_ * y + x] = 0;
          }
        } else {
          std::stack<int> wavefrontIndices;
          wavefrontIndices.push(pixelIndex);
          ++currentLabelIndex;
          regionTypes.push_back(false);
          int regionPixelTotal = 0;
          labels[pixelIndex] = currentLabelIndex;

          while (!wavefrontIndices.empty()) {
            int currentPixelIndex = wavefrontIndices.top();
            wavefrontIndices.pop();
            int currentX = currentPixelIndex % width_;
            int currentY = currentPixelIndex / width_;
            ++regionPixelTotal;
            unsigned short pixelValue = image[width_ * currentY + currentX];

            if (currentX < width_ - 1 && labels[currentPixelIndex + 1] == 0 &&
                image[width_ * currentY + currentX + 1] != 0 &&
                std::abs(pixelValue -
                         image[width_ * currentY + currentX + 1]) <=
                    maxDifference) {
              labels[currentPixelIndex + 1] = currentLabelIndex;
              wavefrontIndices.push(currentPixelIndex + 1);
            }

            if (currentX > 0 && labels[currentPixelIndex - 1] == 0 &&
                image[width_ * currentY + currentX - 1] != 0 &&
                std::abs(pixelValue -
                         image[width_ * currentY + currentX - 1]) <=
                    maxDifference) {
              labels[currentPixelIndex - 1] = currentLabelIndex;
              wavefrontIndices.push(currentPixelIndex - 1);
            }

            if (currentY < height_ - 1 &&
                labels[currentPixelIndex + width_] == 0 &&
                image[width_ * (currentY + 1) + currentX] != 0 &&
                std::abs(pixelValue -
                         image[width_ * (currentY + 1) + currentX]) <=
                    maxDifference) {
              labels[currentPixelIndex + width_] = currentLabelIndex;
              wavefrontIndices.push(currentPixelIndex + width_);
            }

            if (currentY > 0 && labels[currentPixelIndex - width_] == 0 &&
                image[width_ * (currentY - 1) + currentX] != 0 &&
                std::abs(pixelValue -
                         image[width_ * (currentY - 1) + currentX]) <=
                    maxDifference) {
              labels[currentPixelIndex - width_] = currentLabelIndex;
              wavefrontIndices.push(currentPixelIndex - width_);
            }
          }

          if (regionPixelTotal <= maxSpeckleSize) {
            regionTypes[currentLabelIndex] = true;
            image[width_ * y + x] = 0;
          }
        }
      }
    }
  }
}

void SGMFlow::enforceLeftRightConsistency(
    unsigned short *leftVZRatioImage,
    unsigned short *leftplusVZRatioImage) const {
  // Check left disparity image

  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      if (leftVZRatioImage[width_ * y + x] == 0)
        continue;

      int leftDisparityValue = static_cast<int>(
          static_cast<double>(leftVZRatioImage[width_ * y + x]) /
              vzratioFactor_ +
          0.5);
      if (x - leftDisparityValue < 0) {
        leftVZRatioImage[width_ * y + x] = 0;
        continue;
      }

      int rightDisparityValue = static_cast<int>(
          static_cast<double>(
              leftplusVZRatioImage[width_ * y + x - leftDisparityValue]) /
              vzratioFactor_ +
          0.5);
      if (rightDisparityValue == 0 ||
          abs(leftDisparityValue - rightDisparityValue) >
              consistencyThreshold_) {
        leftVZRatioImage[width_ * y + x] = 0;
      }
    }
  }

  // Check right disparity image
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      if (leftplusVZRatioImage[width_ * y + x] == 0)
        continue;

      int rightDisparityValue = static_cast<int>(
          static_cast<double>(leftplusVZRatioImage[width_ * y + x]) /
              vzratioFactor_ +
          0.5);
      if (x + rightDisparityValue >= width_) {
        leftplusVZRatioImage[width_ * y + x] = 0;
        continue;
      }

      int leftDisparityValue = static_cast<int>(
          static_cast<double>(
              leftVZRatioImage[width_ * y + x + rightDisparityValue]) /
              vzratioFactor_ +
          0.5);
      if (leftDisparityValue == 0 ||
          abs(rightDisparityValue - leftDisparityValue) >
              consistencyThreshold_) {
        leftplusVZRatioImage[width_ * y + x] = 0;
      }
    }
  }
}