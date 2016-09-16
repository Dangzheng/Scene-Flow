#pragma once

#include <png++/png.hpp>

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
               float *vzratioImage);

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
  void calcTopRowCost(unsigned char *&leftSobelRow, int *&leftCensusRow,
                      unsigned char *&leftplusSobelRow, int *&leftplusCensusRow,
                      unsigned short *costImageRow);
  void calcPixelwiseSAD(const unsigned char *leftSobelRow,
                        const unsigned char *leftplusSobelRow, int y = 0);
  void calcHalfPixelLeftPlus(const unsigned char *leftplusSobelRow);

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
  unsigned char *halfPixelLeftPlusMin_;
  unsigned char *halfPixelLeftPlusMax_;
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
};