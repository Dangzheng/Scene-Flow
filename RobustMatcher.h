#pragma once

#include <opencv2/legacy/legacy.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

class RobustMatcher {
private:
  cv::Ptr<cv::FeatureDetector> detector;
  cv::Ptr<cv::DescriptorExtractor> extractor;
  float ratio;
  bool refineF;
  double distEpi;
  double confidence;
  int width_;
  float epipoleXCoor, epipoleYCoor;
  int ratioTest(std::vector<std::vector<cv::DMatch>> &matches);
  void symmetryTest(const std::vector<std::vector<cv::DMatch>> &matchesL2LP,
                    const std::vector<std::vector<cv::DMatch>> &matchesLP2L,
                    std::vector<cv::DMatch> &symMatches);
  cv::Mat ransacTest(const std::vector<cv::DMatch> &matches,
                     std::vector<cv::KeyPoint> &keypointsL,
                     std::vector<cv::KeyPoint> &keypointsLP,
                     std::vector<cv::DMatch> &outMatches);
  cv::Mat epipolarGeometryConstraint(std::vector<cv::KeyPoint> &keypointsL,
                                     std::vector<cv::KeyPoint> &keypointsLP,
                                     std::vector<cv::DMatch> &outMatches,
                                     cv::Mat fundamental);
  bool interscetion(cv::Point2f &inter_pt, cv::Point2f o1, cv::Point2f p1,
                    cv::Point2f o2, cv::Point2f p2);
  bool isPointOnSegment(cv::Point2f epipole, cv::Point2f start,
                        cv::Point2f end);

public:
  RobustMatcher()
      : ratio(0.65f), refineF(true), distEpi(0.1), confidence(0.99) {
    // SURF 为默认特征
    detector = new cv::SurfFeatureDetector();
    extractor = new cv::SurfDescriptorExtractor();
  };
  cv::Mat match(cv::Mat &leftImage, cv::Mat &leftplusImage,
                std::vector<cv::DMatch> &matches,
                std::vector<cv::KeyPoint> &keypointsL,
                std::vector<cv::KeyPoint> &keypointsLP);
  float x();
  float y();
  //设置特征检测器
  void setFeatureDetector(cv::Ptr<cv::FeatureDetector> &detect);
  //设置描述子提取器
  void setDescriptorExtractor(cv::Ptr<cv::DescriptorExtractor> &desc);
  void setConfidenceLevel(const float confidencelevel);
  void setMinDistanceToEpipolar(const float mindist);
  void setRatio(const float rate);
};
