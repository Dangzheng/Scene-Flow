#include "SPSFlow.h"
#include "SGMFlow.h"
#include <algorithm>
#include <cmath>
#include <float.h>
#include <stdexcept>

#include <opencv2/opencv.hpp>
// Default parameters
const double SPSFLOW_DEFAULT_OUTPUT_VZ_RATIO =
    256.0; //因为128已经能够满足移动到图像边缘的取值了。
const int SPSFLOW_DEFAULT_OUTER_ITERATION_COUNT = 10;
const int SPSFLOW_DEFAULT_INNER_ITERATION_COUNT = 10;
const double SPSFLOW_DEFAULT_POSITION_WEIGHT = 500.0;
const double SPSFLOW_DEFAULT_VZRATIO_WEIGHT = 2000.0;
const double SPSFLOW_DEFAULT_BOUNDARY_LENGTH_WEIGHT = 1500.0;
const double SPSFLOW_DEFAULT_SMOOTHNESS_WEIGHT = 400.0;
const double SPSFLOW_DEFAULT_INLIER_THRESHOLD = 3.0;
const double SPSFLOW_DEFAULT_HINGE_PENALTY = 5.0;
const double SPSFLOW_DEFAULT_OCCLUSION_PENALTY = 15.0;
const double SPSFLOW_DEFAULT_IMPOSSIBLE_PENALTY = 30.0;
// Pixel offsets of 4- and 8-neighbors

SPSFlow::SPSFlow()
    : outputVZRatio_(SPSFLOW_DEFAULT_OUTPUT_VZ_RATIO),

      outerIterationTotal_(SPSFLOW_DEFAULT_OUTER_ITERATION_COUNT),
      innerIterationTotal_(SPSFLOW_DEFAULT_INNER_ITERATION_COUNT),

      positionWeight_(SPSFLOW_DEFAULT_POSITION_WEIGHT),
      vzratioWeight_(SPSFLOW_DEFAULT_VZRATIO_WEIGHT),
      boundaryLengthWeight_(SPSFLOW_DEFAULT_BOUNDARY_LENGTH_WEIGHT),
      inlierThreshold_(SPSFLOW_DEFAULT_INLIER_THRESHOLD),

      hingePenalty_(SPSFLOW_DEFAULT_HINGE_PENALTY),
      occlusionPenalty_(SPSFLOW_DEFAULT_OCCLUSION_PENALTY),
      impossiblePenalty_(SPSFLOW_DEFAULT_IMPOSSIBLE_PENALTY) {
  smoothRelativeWeight_ =
      SPSFLOW_DEFAULT_SMOOTHNESS_WEIGHT / SPSFLOW_DEFAULT_VZRATIO_WEIGHT;
}
void SPSFlow::setOutputVZRatio(const double outputVZRatio) {
  if (outputVZRatio < 1) {
    throw std::invalid_argument("[SPSFlow::setOutputVZRatio] "
                                "ratio factor is less than 1");
  }

  outputVZRatio_ = outputVZRatio;
}

void SPSFlow::setIterationTotal(const int outerIterationTotal,
                                const int innerIterationTotal) {
  if (outerIterationTotal < 1 || innerIterationTotal < 1) {
    throw std::invalid_argument("[SPSFlow::setIterationTotal] the number of "
                                "iterations is less than 1");
  }
  outerIterationTotal_ = outerIterationTotal;
  innerIterationTotal_ = innerIterationTotal;
}

void SPSFlow::setWeightParameter(const double positionWeight,
                                 const double vzratioWeight,
                                 const double boundaryLengthWeight,
                                 const double smoothnessWeight) {
  if (positionWeight < 0 || vzratioWeight < 0 || boundaryLengthWeight < 0 ||
      smoothnessWeight < 0) {
    throw std::invalid_argument(
        "[SPSFlow::setWeightParameter] weight value is nagative");
  }

  positionWeight_ = positionWeight;
  vzratioWeight_ = vzratioWeight;
  boundaryLengthWeight_ = boundaryLengthWeight;
  smoothRelativeWeight_ = smoothnessWeight / vzratioWeight;
}

void SPSFlow::setInlierThreshold(const double inlierThreshold) {
  if (inlierThreshold <= 0) {
    throw std::invalid_argument("[SPSFlow::setInlierThreshold] threshold of "
                                "inlier is less than zero");
  }

  inlierThreshold_ = inlierThreshold;
}

void SPSFlow::setPenaltyParameter(const double hingePenalty,
                                  const double occlusionPenalty,
                                  const double impossiblePenalty) {
  if (hingePenalty <= 0 || occlusionPenalty <= 0 || impossiblePenalty < 0) {
    throw std::invalid_argument(
        "[SPSFlow::setPenaltyParameter] penalty value is less than zero");
  }
  if (hingePenalty >= occlusionPenalty) {
    throw std::invalid_argument("[SPSFlow::setPenaltyParameter] hinge "
                                "penalty is larger than occlusion penalty");
  }

  hingePenalty_ = hingePenalty;
  occlusionPenalty_ = occlusionPenalty;
  impossiblePenalty_ = impossiblePenalty;
}

// ---------以上确认参数的函数，下面开始计算的函数-----------
void SPSFlow::compute(const int superpixelTotal,
                      const png::image<png::rgb_pixel> &leftImage,
                      const png::image<png::rgb_pixel> &leftplusImage,
                      png::image<png::gray_pixel_16> &segmentImage,
                      png::image<png::gray_pixel_16> &vzratioImage,
                      std::vector<std::vector<double>> &vzratioPlaneParameters,
                      std::vector<std::vector<int>> &boundaryLabels,
                      std::string leftImageFilename,
                      std::string leftplusImageFilename) {
  if (superpixelTotal < 2) {
    throw std::invalid_argument(
        "[SPSFlow::compute] the number of superpixels is less than 2");
  }
  width_ = static_cast<int>(leftImage.get_width());
  height_ = static_cast<int>(leftImage.get_height());
  if (leftplusImage.get_width() != width_ ||
      leftplusImage.get_height() != height_) {
    throw std::invalid_argument("[SPSFlow::setInputData] sizes of left and "
                                "leftplus images are different");
  }

  allocateBuffer();

  setInputData(leftImage, leftplusImage, leftImageFilename,
               leftplusImageFilename);

  freeBuffer();
}
//分配内存和释放内存的代码。
void SPSFlow::allocateBuffer() {
  inputLabImage_ =
      reinterpret_cast<float *>(malloc(width_ * height_ * 3 * sizeof(float)));
  initialVZratioImage_ =
      reinterpret_cast<float *>(malloc(width_ * height_ * sizeof(float)));
  labelImage_ = reinterpret_cast<int *>(malloc(width_ * height_ * sizeof(int)));
  outlierFlagImage_ = reinterpret_cast<unsigned char *>(
      malloc(width_ * height_ * sizeof(unsigned char)));
  boundaryFlagImage_ = reinterpret_cast<unsigned char *>(
      malloc(width_ * height_ * sizeof(unsigned char)));
}

void SPSFlow::freeBuffer() {
  free(inputLabImage_);
  free(initialVZratioImage_);
  free(labelImage_);
  free(outlierFlagImage_);
  free(boundaryFlagImage_);
}

//---------compute 中应用的函数-----------
void SPSFlow::setInputData(const png::image<png::rgb_pixel> &leftImage,
                           const png::image<png::rgb_pixel> &leftplusImage,
                           std::string leftImageFilename,
                           std::string leftplusImageFilename) {
  setLabImage(leftImage);
  //如果是图片的预处理步骤的话为什么只针对左图进行处理呢？
  computeInitialVZratioImage(leftImage, leftplusImage, leftImageFilename,
                             leftplusImageFilename);
}

void SPSFlow::setLabImage(const png::image<png::rgb_pixel> &leftImage) {
  std::vector<float> sRGBGammaCorrections(256);
  for (int pixelValue = 0; pixelValue < 256; ++pixelValue) {
    double normalizedValue = pixelValue / 255.0;
    double transformedValue = (normalizedValue <= 0.04045)
                                  ? normalizedValue / 12.92
                                  : pow((normalizedValue + 0.055) / 1.055, 2.4);
    sRGBGammaCorrections[pixelValue] = static_cast<float>(transformedValue);
    //将256个灰度等级转化到对应的归一化灰度等级上。
    for (int y = 0; y < height_; ++y) {
      for (int x = 0; x < width_; ++x) {
        png::rgb_pixel rgbPixel = leftImage.get_pixel(x, y);

        float correctedR = sRGBGammaCorrections[rgbPixel.red];
        float correctedG = sRGBGammaCorrections[rgbPixel.green];
        float correctedB = sRGBGammaCorrections[rgbPixel.blue];

        float xyzColor[3];
        xyzColor[0] = correctedR * 0.412453f + correctedG * 0.357580f +
                      correctedB * 0.180423f;
        xyzColor[1] = correctedR * 0.212671f + correctedG * 0.715160f +
                      correctedB * 0.072169f;
        xyzColor[2] = correctedR * 0.019334f + correctedG * 0.119193f +
                      correctedB * 0.950227f;

        const double epsilon = 0.008856;
        const double kappa = 903.3;
        const double referenceWhite[3] = {0.950456, 1.0, 1.088754};

        float normalizedX = static_cast<float>(xyzColor[0] / referenceWhite[0]);
        float normalizedY = static_cast<float>(xyzColor[1] / referenceWhite[1]);
        float normalizedZ = static_cast<float>(xyzColor[2] / referenceWhite[2]);
        float fX =
            (normalizedX > epsilon)
                ? static_cast<float>(pow(normalizedX, 1.0 / 3.0))
                : static_cast<float>((kappa * normalizedX + 16.0) / 116.0);
        float fY =
            (normalizedY > epsilon)
                ? static_cast<float>(pow(normalizedY, 1.0 / 3.0))
                : static_cast<float>((kappa * normalizedY + 16.0) / 116.0);
        float fZ =
            (normalizedZ > epsilon)
                ? static_cast<float>(pow(normalizedZ, 1.0 / 3.0))
                : static_cast<float>((kappa * normalizedZ + 16.0) / 116.0);

        inputLabImage_[width_ * 3 * y + 3 * x] =
            static_cast<float>(116.0 * fY - 16.0);
        inputLabImage_[width_ * 3 * y + 3 * x + 1] =
            static_cast<float>(500.0 * (fX - fY));
        inputLabImage_[width_ * 3 * y + 3 * x + 2] =
            static_cast<float>(200.0 * (fY - fZ));
      }
    }
  }
}

void SPSFlow::computeInitialVZratioImage(
    const png::image<png::rgb_pixel> &leftImage,
    const png::image<png::rgb_pixel> &leftplusImage,
    std::string leftImageFilename, std::string leftplusImageFilename) {
  SGMFlow sgm;
  sgm.setVZRatioTotal(128);
  sgm.compute(leftImage, leftplusImage, initialVZratioImage_, leftImageFilename,
              leftplusImageFilename);
}