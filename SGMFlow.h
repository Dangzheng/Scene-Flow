#pragma once

#include <png++/png.hpp>

#include <opencv2/legacy/legacy.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/opencv.hpp>

class SGMFlow {
public:
  SGMFlow();

  void setVZRatioTotal(const int vzratioTotal);
  void setVZRatioFactor(const double vzratioFactor);
  void setDataCostParameters(const int sobelCapValue,
                             const int censusWindowRadius,
                             const double censusWeightFactor,
                             const int aggregationWindowRadius);
  void setSmoothnessCostParameters(const int smoothnessPenaltySmall,
                                   const int smoothnessPenaltyLarge);
  void setConsistencyThreshold(const int consistencyThreshold);

  void compute(const png::image<png::rgb_pixel> &leftImage,
               const png::image<png::rgb_pixel> &leftplusImage,
               float *vzratioImage, std::string leftImageFilename,
               std::string leftplusImageFilename);
  // add by Dangzheng

private:
  void initialize(const png::image<png::rgb_pixel> &leftImage,
                  const png::image<png::rgb_pixel> &leftplusImage);
  void setImageSize(const png::image<png::rgb_pixel> &leftImage,
                    const png::image<png::rgb_pixel> &leftplusImage);
  void allocateDataBuffer();
  void freeDataBuffer();
  void computeCostImage(const png::image<png::rgb_pixel> &leftImage,
                        const png::image<png::rgb_pixel> &leftplusImage);
  void convertToGrayscale(const png::image<png::rgb_pixel> &leftImage,
                          const png::image<png::rgb_pixel> &leftplusImage,
                          unsigned char *leftGrayscaleImage,
                          unsigned char *leftplusGrayscaleImage) const;
  void computeLeftCostImage(const unsigned char *leftGrayscaleImage,
                            const unsigned char *leftplusGrayscaleImage);
  void computeCappedSobelIamge(const unsigned char *image,
                               const bool horizontalFlip,
                               unsigned char *sobelImage) const;
  void computeCensusImage(const unsigned char *image, int *censusImage) const;
  void
  calcTopRowCost(unsigned char *&leftSobelRow, int *&leftCensusRow,
                 unsigned char *&leftplusSobelRow, int *&leftplusCensusRow,
                 unsigned short *costImageRow, unsigned char *&leftSobelImage,
                 unsigned char *&leftplusSobelImage, const int *leftCensusImage,
                 const int *leftplusCensusImage, const bool calcLeft = true);
  void calcRowCosts(unsigned char *&leftSobelRow, int *&leftCensusRow,
                    unsigned char *&leftplusSobelRow, int *&leftplusCensusRow,
                    unsigned short *costImageRow,
                    unsigned char *&leftSobelImage,
                    unsigned char *&leftplusSobelImage,
                    const int *leftCensusImage, const int *leftplusCensusImage,
                    const bool calcLeft = true);
  // add by Dangzheng
  void calcPixelwiseSADHamming(
      const unsigned char *leftSobelRow, const unsigned char *leftplusSobelRow,
      const unsigned char *leftSobelImage,
      const unsigned char *leftplusSobelImage, const int *leftCensusRow,
      const int *leftplusCensusRow, const int *leftCensusImage,
      const int *leftplusCensusImage, const int yIndex = 0,
      const bool calcLeft = true);
  void addPixelwiseHamming(const int *leftCensusRow,
                           const int *leftplusCensusRow,
                           const int *leftCensusImage,
                           const int *leftplusCensusImage,
                           const bool calcLeft = true, const int xBase = 0,
                           const int xMatching = 0, const int yMatching = 0,
                           const int wp = 0);
  //->
  void calcHalfPixelLeftPlus(const unsigned char *leftplusSobelRow);
  void addPixelwiseHamming(const int *leftCensusRow,
                           const int *leftplusCensusRow);
  void computeLeftPlusCostImage();
  void performSGM(unsigned short *costImage, unsigned short *vzratioImage);
  void speckleFilter(const int maxSpeckleSize, const int maxDifference,
                     unsigned short *image) const;
  void enforceLeftRightConsistency(unsigned short *leftVZRatioImage,
                                   unsigned short *leftplusVZRatioImage) const;

  // Parameters(Parameters
  // 和Data这些变量名字的定义都是直接照搬的，出现问题再考虑是什么问题。)
  int vzratioTotal_;
  double vzratioFactor_;
  int sobelCapValue_;
  int censusWindowRadius_;
  double censusWeightFactor_;
  int aggregationWindowRadius_;
  int smoothnessPenaltySmall_;
  int smoothnessPenaltyLarge_;
  int consistencyThreshold_;
  // Data
  int width_;
  int height_;
  int widthStep_;
  unsigned short *leftCostImage_;
  unsigned short *leftplusCostImage_;
  unsigned char *pixelwiseCostRow_;
  unsigned short *rowAggregatedCost_;

  int pathRowBufferTotal_;
  int vzratioSize_;
  int pathTotal_;
  int pathVZratioSize_;
  int costSumBufferRowSize_;
  int costSumBufferSize_;
  int pathMinCostBufferSize_;
  int pathCostBufferSize_;
  int totalBufferSize_;
  short *sgmBuffer_;

  // add by Dangzheng
  void calcEpipoleRotaionVector(std::string leftImageFilename,
                                std::string leftplusImageFilename);

  double wx_t;
  double wy_t;
  double wz_t;
  double wx_inv;
  double wy_inv;
  double wz_inv;
  double f = 707.091200;
  double epipoleX;
  double epipoleY;

  // 几个censusImage和sobelImage
  unsigned char *leftSobelImage;
  unsigned char *leftplusSobelImage;
  int *leftCensusImage;
  int *leftplusCensusImage;
};