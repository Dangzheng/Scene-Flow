#include "SGMFlow.h"
#include <algorithm>
#include <math.h>
#include <nmmintrin.h>
#include <stack>
#include <stdexcept>

#include <vector>

#include <opencv2/opencv.hpp>
// Default parameters
const int SGMFLOW_DEFAULT_VZRATIO_TOTAL = 256;
const double SGMFLOW_DEFAULT_VZRATIO_FACTOR = 256;
const int SGMFLOW_DEFAULT_SOBEL_CAP_VALUE = 15;
const int SGMFLOW_DEFAULT_CENSUS_WINDOW_RADIUS = 2;
const double SGMFLOW_DEFAULT_CENSUS_WEIGHT_FACTOR = 1.0 / 6.0;
const int SGMFLOW_DEFAULT_AGGREGATION_WINDOW_RADIUS = 2;
const int SGMFLOW_DEFAULT_SMOOTHNESS_PENALTY_SMALL = 100;
const int SGMFLOW_DEFAULT_SMOOTHNESS_PENALTY_LARGE = 1600;
const int SGMFLOW_DEFAULT_CONSISTENCY_THRESHOLD = 1;

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
//------------------------------------------------------------------------
void SGMFlow::compute(const png::image<png::rgb_pixel> &leftImage,
                      const png::image<png::rgb_pixel> &leftplusImage,
                      float *vzratioImage) {
  initialize(leftImage, leftplusImage);

  computeCostImage(leftImage, leftplusImage);
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
  int halfPixelLeftPlusBufferSize = widthStep_;

  pixelwiseCostRow_ = reinterpret_cast<unsigned char *>(
      _mm_malloc(pixelwiseCostRowBufferSize * sizeof(unsigned char), 16));
  rowAggregatedCost_ = reinterpret_cast<unsigned short *>(
      _mm_malloc(rowAggregatedCostBufferSize * sizeof(unsigned short), 16));
  halfPixelLeftPlusMin_ = reinterpret_cast<unsigned char *>(
      _mm_malloc(halfPixelLeftPlusBufferSize * sizeof(unsigned char), 16));
  halfPixelLeftPlusMax_ = reinterpret_cast<unsigned char *>(
      _mm_malloc(halfPixelLeftPlusBufferSize * sizeof(unsigned char), 16));

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
  _mm_free(halfPixelLeftPlusMin_);
  _mm_free(halfPixelLeftPlusMax_);
  _mm_free(sgmBuffer_);
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
  computeLeftCostImage(leftGrayscaleImage, leftplusGrayscaleImage);
  //这个函数是执行SGM算法的核心函数，如果想要把Stereo改成flow那么应该在这个位置将极线几何部分的变换修改一下就行了。
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
}

void SGMFlow::computeLeftCostImage(
    const unsigned char *leftGrayscaleImage,
    const unsigned char *leftplusGrayscaleImage) {
  unsigned char *leftSobelImage = reinterpret_cast<unsigned char *>(
      _mm_malloc(widthStep_ * height_ * sizeof(unsigned char), 16));
  unsigned char *leftplusSobelImage = reinterpret_cast<unsigned char *>(
      _mm_malloc(widthStep_ * height_ * sizeof(unsigned char), 16));
  //此处用sobel算子来处理图像，sobel处理完的图像留下的都是图像的边缘，对左图像检测水平方向的边缘，对左plus检测垂直方向的边缘。
  computeCappedSobelIamge(leftGrayscaleImage, false, leftSobelImage);
  computeCappedSobelIamge(leftplusGrayscaleImage, true, leftplusSobelImage);

  int *leftCensusImage =
      reinterpret_cast<int *>(malloc(width_ * height_ * sizeof(int)));
  int *leftplusCensusImage =
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
                 leftplusCensusRow, costImageRow);
  //将sobel处理过的图片，census Image送到函数里面计算匹配代价。
}

void SGMFlow::computeCappedSobelIamge(const unsigned char *image,
                                      const bool horizontalFlip,
                                      unsigned char *sobelImage) const {
  memset(sobelImage, sobelCapValue_, widthStep_ * height_);
  //将sobelImage所在内存的前widthStep_ *
  // height_个字节，用sobelCapValue_的值来进行替换。
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

void SGMFlow::calcTopRowCost(unsigned char *&leftSobelRow, int *&leftCensusRow,
                             unsigned char *&leftplusSobelRow,
                             int *&leftplusCensusRow,
                             unsigned short *costImageRow) {
  //这个函数标注的计算顶行的cost，顶行因为在边缘，所以要特殊处理。
  for (int rowIndex = 0; rowIndex <= aggregationWindowRadius_; ++rowIndex) {
    //都是在计算不超过窗的半径的cost。
    int rowAggregatedCostIndex =
        std::min(rowIndex, height_ - 1) % (aggregationWindowRadius_ * 2 + 2);
    //行数和最大行数两者之前取个最小，这个是防止程序出错做的防护。
    unsigned short *rowAggregatedCostCurrent =
        rowAggregatedCost_ + rowAggregatedCostIndex * width_ * vzratioTotal_;
    //这个rowAggregatedCost_是哪里来的?算了，计算SAD的时候也没有用到那就先搁置不管了。
    calcPixelwiseSAD(leftSobelRow, leftplusSobelRow, 0);
  }
}

void SGMFlow::calcPixelwiseSAD(const unsigned char *leftSobelRow,
                               const unsigned char *leftplusSobelRow, int y) {
  //这个函数的标注是计算像素级别的SAD cost。
  //如果是计算顶行的话，那么y就设置成0
  calcHalfPixelLeftPlus(leftplusSobelRow); //为什么要用sobel来计算半像素？

  double wx_t = 0.22621486;
  double wy_t = 0.061689742;
  double wz_t = 1.3907996;

  double f = 721.53770;
  double cx = 609.559300;
  double cy = 172.854000;

  double epipoleX = 605.20;
  double epipoleY = 168.78;
  for (int x = 0; x < width_; ++x) {
    int leftCenterValue = leftSobelRow[x];
    int leftHalfLeftValue =
        x > 0 ? (leftCenterValue + leftSobelRow[x - 1]) / 2 : leftCenterValue;
    int leftHalfRightValue = x < width_ - 1
                                 ? (leftCenterValue + leftSobelRow[x + 1]) / 2
                                 : leftCenterValue;
    int leftMinValue = std::min(leftHalfLeftValue, leftHalfRightValue);
    leftMinValue = std::min(leftMinValue, leftCenterValue);
    //在左边插值出来的像素和右边插值出来的像素，以及自己本身三个像素中选择一个。
    int leftMaxValue = std::min(leftHalfLeftValue, leftHalfRightValue);
    leftMaxValue = std::max(leftMaxValue, leftCenterValue);
    // 作者的代码在这个位置用了一些技巧，将每行计算的cost存储在pixelwiseCostRow_之中。
    // 因为Stereo的搜索极线都是水平的，并且y坐标是相同的，但是在flow中虽然极线可能是水平
    // 的，但是，两幅图片中极线的y坐标不一定相同。
    // 这个地方之所以给人用left左边缘去匹配leftplus的右边缘，是因为之前计算sobel的时候，
    // leftplus是被左右翻转了的。
    // 再多说一些这里的实现思路:首先，计算极点的时候我是直接在图像上面计算的极点，就是两条
    // 极线的交点，所以这个坐标直接就是极线的坐标。延极线方向的移动只需要目标点坐标减去极
    // 点坐标求得单位方向向量之后，就可以沿着极线方向运动，他的那个d注意看，是leftplus减
    // left的距离，所以如果d是的正的话，他是背离极点运动的，所以一定要以极点作为起点。
    // 这里面有一个trick：在文章中的推导是在相机坐标系下，d不是（u,v）上面的像素坐标平移，所以要乘以焦距。
    // 第二点，这个代码是参照双目立体视觉匹配写的，所以他在写的时候就是一行一行计算，因为本
    // 身双目立体视觉的极线就是平行于x轴的，但是光流里面极线不平行，暂时按照极线几何的路数
    // 走把。
    //此处vmax还不能确定应该用什么样的方式确定合理，以后再将其写成参数设置，此处仅仅将其写为
    //一个变量先定义一下，用作测试吧
    float vMax = 1 / 0.8;
    int n = 256;
    //此处的0.8指的是距离相机光心，可以测量的最小的深度值为0.8，但是理论上来讲，成像平面在
    // z= 1处，所以最小的能测量的深度应该是1。

    for (int wp = 0; wp <= vzratioTotal_; ++wp) {
      // 这里的d应该是指的disparity，将其改写成vzratio。
      //
      //这个取值有问题，因为flow、应该是先计算位置，然后再进行取值，这个地方如果取整数值的话，
      //这个点不一定在极线上面，讲道理这里为了保证这个点在极线上应该进行一个插值处理，但是现在
      //先按照整数来取值
      double uwTransX, uwTransY;
      // calculate the rotation
      uwTransX = (x - cx) +
                 (f * wx_t - wz_t * y + wy_t * x * x / f - wx_t * x * y / f);
      uwTransY = (y - cy) +
                 (-f * wx_t + wz_t * x + wy_t * x * y / f - wx_t * y * y / f);
      // distance between p and epipole o;
      double distancePEpi = sqrt(pow(x + uwTransX - epipoleX, 2) +
                                 pow(y + uwTransY - epipoleY, 2));

      double distanceRRhat =
          distancePEpi * wp * (vMax / n) / (1 - wp * (vMax / n));
      std::cout << "reach here! dis" << std::endl;

      std::cout << "distanceRRhat: " << distanceRRhat << std::endl;
      //这个位置面临的问题是，当vzration不断增加时，其背离极点的距离越来越远，可能会超出图像的边缘
      //那么这个时候，就会报错。在立体视觉问题中这个很好判断，因为只要给横坐标上限制就可以了，但是现
      //在面临的问题是，如何提出一个泛化的限制。
      //还存在的一个问题就是：提出了限制之后，剩余灰度等级的代价该如何填充。
      double directionX = x - epipoleX;
      double directionY = y - epipoleY;
      directionX = directionX / sqrt(pow(directionX, 2) + pow(directionY, 2));
      directionY = directionY / sqrt(pow(directionX, 2) + pow(directionY, 2));
      int xPlus = x + uwTransX + directionX * distanceRRhat;
      int yPlus = y + uwTransY + directionY * distanceRRhat;
      // 现在求出了根据VZ-ratio计算的，left图像上一个点，在leftplus图像上的对应点。
      // 这个点式计算出来的理论值点，并不是真真切切在计算的sobelImage上面的点。
      // 这个位置先看看计算结果，如果不好的话，以后可能还要给一个插值，现在先给一个向下取整
      // widthStep_ * y + width_ - x
      std::cout << uwTransX + directionX * distanceRRhat << std::endl;
      std::cout << "(" << x << "," << y << ")" << std::endl;
      int leftplusCenterValue =
          leftplusSobelRow[widthStep_ * yPlus + width_ - xPlus - 1];
      int leftplusMinValue =
          halfPixelLeftPlusMin_[widthStep_ * yPlus + width_ - xPlus - 1];
      int leftplusMaxValue =
          halfPixelLeftPlusMax_[widthStep_ * yPlus + width_ - xPlus - 1];
      // 可以看到这个位置取点直接就是按位置取点
      int costLtoR = std::max(0, leftCenterValue - leftplusMaxValue);
      costLtoR = std::max(costLtoR, leftplusMinValue - leftCenterValue);
      int costRtoL = std::max(0, leftplusCenterValue - leftMaxValue);
      costRtoL = std::max(costRtoL, leftMinValue - leftplusCenterValue);
      int costValue = std::min(costLtoR, costRtoL);
      //上面的部分是没有考量过边界条件和优化的一个cost计算
      //计算完cost之后将其存贮
      pixelwiseCostRow_[vzratioTotal_ * x + wp] = costValue;
      std::cout << "wp = " << wp << "///costValue: " << costValue << std::endl;
    }
    std::cout << "reach here! Finish on point cost calculation" << std::endl;
    cv::waitKey(0);
  }
}

void SGMFlow::calcHalfPixelLeftPlus(const unsigned char *leftplusSobelRow) {
  for (int x = 0; x < width_; ++x) {
    int centerValue = leftplusSobelRow[x];
    int leftHalfValue =
        x > 0 ? (centerValue + leftplusSobelRow[x - 1]) / 2 : centerValue;
    int rightHalfValue = x < width_ - 1
                             ? (centerValue + leftplusSobelRow[x + 1]) / 2
                             : centerValue;
    //用左边的点计算一次halfvalue，右边点计算一次，然后统计他们的最大最小值。
    //有什么用呢？
    int minValue = std::min(leftHalfValue, rightHalfValue);
    minValue = std::min(minValue, centerValue);
    int maxValue = std::max(leftHalfValue, rightHalfValue);
    maxValue = std::max(maxValue, centerValue);

    halfPixelLeftPlusMin_[x] = minValue;
    halfPixelLeftPlusMax_[x] = maxValue;
  }
}