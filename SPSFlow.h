#pragma once
#include <png++/png.hpp>
#include <stack>
#include <string>
#include <vector>

class SPSFlow {
public:
  SPSFlow();
  void setOutputVZRatio(const double outputVZRatio);
  void setIterationTotal(const int outerIterationTotal,
                         const int innerIterationTotal);
  void setWeightParameter(const double positionWeight,
                          const double vzratioWeight,
                          const double boundaryLengthWeight,
                          const double smoothnessWeight);
  void setInlierThreshold(const double inlierThreshold);
  void setPenaltyParameter(const double hingePenalty,
                           const double occlusionPenalty,
                           const double impossiblePenalty);

  void compute(const int superpixelTotal,
               const png::image<png::rgb_pixel> &leftImage,
               const png::image<png::rgb_pixel> &leftplusImage,
               png::image<png::gray_pixel_16> &segmentImage,
               png::image<png::gray_pixel_16> &vzratioImage,
               std::vector<std::vector<double>> &vzratioPlaneParameters,
               std::vector<std::vector<int>> &boundaryLabels);

private:
  void allocateBuffer();
  void freeBuffer();

  void setInputData(const png::image<png::rgb_pixel> &leftImage,
                    const png::image<png::rgb_pixel> &leftplusImage);
  void setLabImage(const png::image<png::rgb_pixel> &leftImage);
  void
  computeInitialVZratioImage(const png::image<png::rgb_pixel> &leftImage,
                             const png::image<png::rgb_pixel> &leftplusImage);
  // Parameter
  double outputVZRatio_;

  int outerIterationTotal_;
  int innerIterationTotal_;

  double positionWeight_;
  double vzratioWeight_;
  double boundaryLengthWeight_;
  double smoothRelativeWeight_;

  double inlierThreshold_;

  double hingePenalty_;
  double occlusionPenalty_;
  double impossiblePenalty_;

  // Input data
  int width_;
  int height_;
  float *inputLabImage_;
  float *initialVZratioImage_;

  // Superpixel segments;
  int *labelImage_;
  unsigned char *outlierFlagImage_;
  unsigned char *boundaryFlagImage_;
};