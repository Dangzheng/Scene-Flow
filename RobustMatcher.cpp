#include "RobustMatcher.h"
#include <iostream>
void RobustMatcher::setFeatureDetector(cv::Ptr<cv::FeatureDetector> &detect) {
  detector = detect;
}

//设置描述子提取器
void RobustMatcher::setDescriptorExtractor(
    cv::Ptr<cv::DescriptorExtractor> &desc) {
  extractor = desc;
}

void RobustMatcher::setConfidenceLevel(const float confidencelevel) {
  confidence = confidencelevel;
}

void RobustMatcher::setMinDistanceToEpipolar(const float mindist) {
  distEpi = mindist;
}

void RobustMatcher::setRatio(const float rate) { ratio = rate; }

float RobustMatcher::x() { return epipoleXCoor; }

float RobustMatcher::y() { return epipoleYCoor; }

cv::Mat RobustMatcher::match(cv::Mat &leftImage, cv::Mat &leftplusImage,
                             std::vector<cv::DMatch> &matches,
                             std::vector<cv::KeyPoint> &keypointsL,
                             std::vector<cv::KeyPoint> &keypointsLP) {
  width_ = leftImage.cols;
  // 1/检测SURF特征
  detector->detect(leftImage, keypointsL);
  detector->detect(leftplusImage, keypointsLP);
  cv::Mat descriptorsL, descriptorsLP;
  extractor->compute(leftImage, keypointsL, descriptorsL);
  extractor->compute(leftplusImage, keypointsLP, descriptorsLP);

  // 2/匹配两幅图像的描述子
  // cv::BruteForceMatcher<cv::L2<float>> matcher;
  cv::BFMatcher matcher(cv::NORM_L2, false);
  // t->t+1
  std::vector<std::vector<cv::DMatch>> matchesL2LP, matchesLP2L;
  matcher.knnMatch(descriptorsL, descriptorsLP, matchesL2LP, 2);
  matcher.knnMatch(descriptorsLP, descriptorsL, matchesLP2L, 2);

  // 3/移除NN比率大于阈值的匹配

  int removed = ratioTest(matchesL2LP);
  removed = ratioTest(matchesLP2L);

  // 4/移除非对称的匹配
  std::vector<cv::DMatch> symMatches;
  symmetryTest(matchesL2LP, matchesLP2L, symMatches);

  if (symMatches.size() == 0) {
    std::cerr << "Not enough matches." << std::endl;
    cv::Mat fundamental;
    return (fundamental);
  }

  // 5/使用RANSAC进行最终的验证
  cv::Mat fundamental =
      ransacTest(symMatches, keypointsL, keypointsLP, matches);

  cv::Mat fundamentalMat =
      epipolarGeometryConstraint(keypointsL, keypointsLP, matches, fundamental);
  return fundamentalMat;
}

int RobustMatcher::ratioTest(std::vector<std::vector<cv::DMatch>> &matches) {
  int removed = 0;
  //遍历所有的匹配
  for (std::vector<std::vector<cv::DMatch>>::iterator matchIterator =
           matches.begin();
       matchIterator != matches.end(); ++matchIterator) {
    //如果识别两个最近邻
    if (matchIterator->size() > 1) {
      //检查距离比率
      if ((*matchIterator)[0].distance / (*matchIterator)[1].distance > ratio) {
        matchIterator->clear(); //移除匹配
        removed++;
      }
    } else {
      //不包含两个最近邻
      matchIterator->clear();
      removed++;
    }
  }

  return removed;
}

void RobustMatcher::symmetryTest(
    const std::vector<std::vector<cv::DMatch>> &matchesL2LP,
    const std::vector<std::vector<cv::DMatch>> &matchesLP2L,
    std::vector<cv::DMatch> &symMatches) {
  //遍历t->t+1的所有匹配

  for (std::vector<std::vector<cv::DMatch>>::const_iterator matchIterator1 =
           matchesL2LP.begin();
       matchIterator1 != matchesL2LP.end(); ++matchIterator1) {
    //忽略被删除的匹配
    if (matchIterator1->size() < 2)
      continue;

    for (std::vector<std::vector<cv::DMatch>>::const_iterator matchIterator2 =
             matchesLP2L.begin();
         matchIterator2 != matchesLP2L.end(); ++matchIterator2) {
      //忽略被删除的匹配
      if (matchIterator2->size() < 2)
        continue;

      if ((*matchIterator1)[0].queryIdx == (*matchIterator2)[0].trainIdx &&
          (*matchIterator2)[0].queryIdx == (*matchIterator1)[0].trainIdx) {
        //添加对称的匹配
        symMatches.push_back(cv::DMatch((*matchIterator1)[0].queryIdx,
                                        (*matchIterator1)[0].trainIdx,
                                        (*matchIterator1)[0].distance));
        break;
      }
    }
  }
}

//基于RANSAC识别优质匹配
//返回基础矩阵
cv::Mat RobustMatcher::ransacTest(const std::vector<cv::DMatch> &matches,
                                  std::vector<cv::KeyPoint> &keypointsL,
                                  std::vector<cv::KeyPoint> &keypointsLP,
                                  std::vector<cv::DMatch> &outMatches) {
  std::vector<cv::Point2f> points1, points2;
  for (std::vector<cv::DMatch>::const_iterator it = matches.begin();
       it != matches.end(); ++it) {
    //得到左边特征点的坐标
    float x = keypointsL[it->queryIdx].pt.x;
    float y = keypointsL[it->queryIdx].pt.y;
    points1.push_back(cv::Point2f(x, y));
    //右边
    x = keypointsLP[it->trainIdx].pt.x;
    y = keypointsLP[it->trainIdx].pt.y;
    points2.push_back(cv::Point2f(x, y));
  }
  //基于RANSAC计算F矩阵
  std::vector<uchar> inliers(points1.size(), 0);
  cv::Mat fundamental = cv::findFundamentalMat(points1, points2, inliers,
                                               CV_FM_LMEDS, 3, confidence);
  //提取通过的匹配
  std::vector<uchar>::const_iterator itIn = inliers.begin();
  std::vector<cv::DMatch>::const_iterator itM = matches.begin();
  //遍历所有的匹配
  for (; itIn != inliers.end(); ++itIn, ++itM) {
    if (*itIn) {
      //有效匹配
      outMatches.push_back(*itM);
    }
  }
  //
  // std::nth_element(outMatches.begin(), outMatches.begin() + 24,
  //                  outMatches.end());
  // outMatches.erase(outMatches.begin() + 20, outMatches.end());
  // cv::Mat imageMatches;
  // cv::Mat leftImage, leftplusImage;
  // leftImage = cv::imread("000148_10.png", 1);
  // leftplusImage = cv::imread("000148_11.png", 1);
  // cv::drawMatches(leftImage, keypointsL, leftplusImage, keypointsLP,
  // outMatches,
  //                 imageMatches, cv::Scalar(0, 255, 0));
  // cv::imshow("feature", imageMatches);
  // cv::waitKey(0);
  // std::cout << "Number of matched points (after cleaning): "
  //           << outMatches.size() << std::endl;

  if (refineF) {
    // F矩阵将使用所有接受的匹配重新计算
    //转化KeyPoint类型到Point2f
    //准备计算最终的F矩阵
    points1.clear();
    points2.clear();
    for (std::vector<cv::DMatch>::const_iterator it = outMatches.begin();
         it != outMatches.end(); ++it) {
      float x = keypointsL[it->queryIdx].pt.x;
      float y = keypointsL[it->queryIdx].pt.y;
      points1.push_back(cv::Point2f(x, y));
      x = keypointsLP[it->trainIdx].pt.x;
      y = keypointsLP[it->trainIdx].pt.y;
      points2.push_back(cv::Point2f(x, y));
    }
    fundamental = cv::findFundamentalMat(cv::Mat(points1), cv::Mat(points2),
                                         CV_FM_LMEDS, 3, confidence);
  }

  return fundamental;
}

cv::Mat RobustMatcher::epipolarGeometryConstraint(
    std::vector<cv::KeyPoint> &keypointsL,
    std::vector<cv::KeyPoint> &keypointsLP, std::vector<cv::DMatch> &outMatches,
    cv::Mat fundamental) {
  //我这里就用一幅图片进行极线几何校正。
  //总体的思路就是利用在右图中的极线两两成对然后生成极点，如果有些点与均值相差太大，就将这些点抛弃
  //作者的思想是利用极线几何对参数进行重新的估计，我觉得也是可行的，那需要做的相应的修改：
  // 1、将之前的内点改为全部的点，然后
  cv::vector<cv::Vec3f> lines[2];
  cv::vector<cv::Point2f> points1, points2;
  cv::Point2f pt;

  for (std::vector<cv::DMatch>::const_iterator it = outMatches.begin();
       it != outMatches.end(); ++it) {
    float x = keypointsL[it->queryIdx].pt.x;
    float y = keypointsL[it->queryIdx].pt.y;
    points1.push_back(cv::Point2f(x, y));
    x = keypointsLP[it->trainIdx].pt.x;
    y = keypointsLP[it->trainIdx].pt.y;
    points2.push_back(cv::Point2f(x, y));
  }

  cv::Mat imgpt[2];
  imgpt[0] = cv::Mat(points1);
  imgpt[1] = cv::Mat(points2);
  for (int k = 0; k < 2; k++) {
    cv::computeCorrespondEpilines(imgpt[k], k + 1, fundamental, lines[k]);
    if (k == 1) {
      cv::Point2f interPt, o0, o1, p0, p1;
      cv::Mat epipoleMeanX, epipoleMeanY, epipoleVarX, epipoleVarY;
      cv::Mat epipoleX(outMatches.size() - 1, 1, CV_32F);
      cv::Mat epipoleY(outMatches.size() - 1, 1, CV_32F);
      int count = 0;
      for (std::vector<cv::Vec3f>::const_iterator it = lines[k].begin();
           it != lines[k].end(); count++) {
        o0.x = 0;
        o0.y = -(*it)[2] / (*it)[1];
        p0.x = width_;
        p0.y = -((*it)[2] + (*it)[0] * width_) / (*it)[1];
        it++;
        o1.x = 0;
        o1.y = -(*it)[2] / (*it)[1];
        p1.x = width_;
        p1.y = -((*it)[2] + (*it)[0] * width_) / (*it)[1];
        interscetion(interPt, o0, p0, o1, p1);
        epipoleX.at<float>(count, 0) = interPt.x;
        epipoleY.at<float>(count, 0) = interPt.y;
      }

      cv::meanStdDev(epipoleX, epipoleMeanX, epipoleVarX);
      cv::meanStdDev(epipoleY, epipoleMeanY, epipoleVarY);

      cv::Mat tempX, tempY;
      for (int i = 0; i < epipoleX.rows; i++) {
        if (fabs(epipoleX.at<float>(i, 0) - epipoleMeanX.at<double>(0, 0)) <
            distEpi) {
          tempX.push_back(epipoleX.at<float>(i, 0));
        }
        if (fabs(epipoleY.at<float>(i, 0) - epipoleMeanY.at<double>(0, 0)) <
            distEpi) {
          tempY.push_back(epipoleY.at<float>(i, 0));
        }
      }
      swap(epipoleX, tempX);
      swap(epipoleY, tempY);
      meanStdDev(epipoleX, epipoleMeanX, epipoleVarX);
      meanStdDev(epipoleY, epipoleMeanY, epipoleVarY);
      std::cout << "epipole coordinate: (" << epipoleMeanX << ","
                << epipoleMeanY << ")" << std::endl;
      std::cout << "epipole coordinate variance:" << epipoleVarX << "/"
                << epipoleVarY << std::endl;

      epipoleXCoor = epipoleMeanX.at<double>(0, 0);
      epipoleYCoor = epipoleMeanY.at<double>(0, 0);
      //用这个函数来去除match中不符合极线限制的点。
      cv::Point2f epipole;
      epipole.x = epipoleMeanX.at<double>(0, 0);
      epipole.y = epipoleMeanY.at<double>(0, 0);
      std::vector<cv::Vec3f>::const_iterator itL = lines[k].begin();
      std::vector<cv::DMatch>::const_iterator itOM = outMatches.begin();
      points1.clear();
      points2.clear();
      for (; itL != lines[k].end(); ++itL, ++itOM) {
        o0.x = 0;
        o0.y = -(*itL)[2] / (*itL)[1];
        p0.x = width_;
        p0.y = -((*itL)[2] + (*itL)[0] * width_) / (*itL)[1];

        if (isPointOnSegment(epipole, o0, p0)) {
          float x = keypointsL[itOM->queryIdx].pt.x;
          float y = keypointsL[itOM->queryIdx].pt.y;
          points1.push_back(cv::Point2f(x, y));
          x = keypointsLP[itOM->trainIdx].pt.x;
          y = keypointsLP[itOM->trainIdx].pt.y;
          points2.push_back(cv::Point2f(x, y));
        }
      }
    }
  }
  // here clc the final fundamental;

  fundamental = cv::findFundamentalMat(cv::Mat(points1), cv::Mat(points2),
                                       CV_FM_LMEDS, 3, confidence);
  // minimizing the sum of epipole constraint for all pixel.

  return fundamental;
}

bool RobustMatcher::isPointOnSegment(cv::Point2f epipole, cv::Point2f start,
                                     cv::Point2f end) {
  float x1 = epipole.x;
  float y1 = epipole.y;
  float x2 = start.x;
  float y2 = start.y;
  float x3 = end.x;
  float y3 = end.y;
  if (x1 * y2 + x3 * y1 + x2 * y3 - x3 * y2 - x2 * y1 - x1 * y3 > 0.1) {
    return false;
  } else {
    return true;
  }
}

bool RobustMatcher::interscetion(cv::Point2f &inter_pt, cv::Point2f o1,
                                 cv::Point2f p1, cv::Point2f o2,
                                 cv::Point2f p2) {
  cv::Point2f x = o2 - o1;
  cv::Point2f d1 = p1 - o1;
  cv::Point2f d2 = p2 - o2;

  float cross = d1.x * d2.y - d1.y * d2.x;
  if (std::abs(cross) < /*EPS*/ 1e-8)
    return false;

  double t1 = (x.x * d2.y - x.y * d2.x) / cross;
  inter_pt = o1 + d1 * t1;
  return true;
};
