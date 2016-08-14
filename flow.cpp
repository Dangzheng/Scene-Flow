#include <vector>
//#include <fstream>
#include <opencv2/opencv.hpp>
#include <cv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>

using namespace cv;
int main(int argc,char* argv[]){
    
    if(argc < 3){
        std::cerr <<"usage: flow left_0 left_1"<<std::endl;
        exit(1);
    }
    
    Mat leftImage0,leftImage1;
    leftImage0 = imread(argv[1],1);
    leftImage1 = imread(argv[2],1);
    
    //imshow("Flow_image_0",leftImage0);
    //imshow("Flow_image_1",leftImage1);
    //waitKey(0);

    //SIFT
    SiftFeatureDetector siftdtc;
    vector<KeyPoint>kp0,kp1;
    Mat Keypoint0,Keypoint1;
    
    siftdtc.detect(leftImage0, kp0);
    //drawKeypoints(leftImage0, kp0, Keypoint0);
    //imshow("leftImage0_keypoints",Keypoint0);
    //waitKey(0);
    
    /*
    vector<KeyPoint>::iterator itvc;
    
    for(itvc=kp0.begin();itvc!=kp0.end();itvc++)
    {
        std::cout<<"angle:"<<itvc->angle<<"\t"<<itvc->class_id<<"\t"<<itvc->octave<<"\t"<<itvc->pt<<"\t"<<itvc->response<<std::endl;
    }
    */
    //angle表示特征点的方向，负值表示不使用
    //class_id表示聚类的ID
    //pt 表示坐标
    
    siftdtc.detect(leftImage1, kp1);
    //drawKeypoints(leftImage1, kp1, Keypoint1);
    
    SiftDescriptorExtractor extractor;
    Mat descriptor0, descriptor1;
    BruteForceMatcher< L2<float> > matcher;
    vector<DMatch> matches;
    Mat MatchesImage;
    extractor.compute(leftImage0, kp0, descriptor0);
    extractor.compute(leftImage1, kp1, descriptor1);
    
    //imshow("descriptor",descriptor0);
    //waitKey(0);
    
    //std::cout<<std::endl<<descriptor0<<std::endl;
    matcher.match(descriptor0, descriptor1, matches);
    drawMatches(leftImage0, kp0, leftImage0, kp1, matches, MatchesImage);
    imshow("Match",MatchesImage);
    waitKey(0);
    
    
    //LMeds或者8-points估计fundamental matrix
    
    //kp0,kp1为feature points需要将其转换array
    int ptCount = matches.size();
    vector<Point2f> points0(ptCount);
    vector<Point2f> points1(ptCount);
    Point2f pt;
    for(int i=0; i<ptCount; i++){
        pt = kp0[matches[i].queryIdx].pt;
        points0[i].x = kp0[i].pt.x;
        points0[i].y = kp0[i].pt.y;
        pt = kp1[matches[1].trainIdx].pt;
        points1[i].x = kp1[i].pt.x;
        points1[i].y = kp1[i].pt.y;
    }
    
    //根据第一次的计算结果去除outlier和误差大的点。
    Mat FundamentalMatrix;
    vector<uchar> LMedsStatus;
    FundamentalMatrix = findFundamentalMat(points0, points1, FM_LMEDS, 3, 0.99, LMedsStatus);
    int Outliers = 0;
    for(int i = 0;i<ptCount;i++){
        if (LMedsStatus[i]==0){
            Outliers++;
        }
    }
    std::cout<<Outliers<<std::endl;
    //根据点的status来去除outlier。
    
    vector<Point2f> pt0_inlier;
    vector<Point2f> pt1_inlier;
    vector<DMatch> InlierMatches;
    int InlierCount = ptCount - Outliers;
    InlierMatches.resize(InlierCount);
    pt0_inlier.resize(InlierCount);
    pt1_inlier.resize(InlierCount);
    InlierCount = 0;
    for(int i=0;i<ptCount;i++){
        if(LMedsStatus[i]!=0){
            pt0_inlier[InlierCount].x = points0[i].x;
            pt0_inlier[InlierCount].y = points0[i].y;
            pt1_inlier[InlierCount].x = points1[i].x;
            pt1_inlier[InlierCount].y = points1[i].y;
            InlierMatches[InlierCount].queryIdx = InlierCount;
            InlierMatches[InlierCount].trainIdx = InlierCount;
            InlierCount++;
        }
    }
    
    FundamentalMatrix = findFundamentalMat(pt0_inlier, pt1_inlier, FM_LMEDS, 3, 0.99);
    
    vector<KeyPoint> key0(InlierCount);
    vector<KeyPoint> key1(InlierCount);
    KeyPoint::convert(pt0_inlier, key0);
    KeyPoint::convert(pt1_inlier, key1);
    drawMatches(leftImage0, key0, leftImage0, key1, InlierMatches, MatchesImage);
    imshow("Match",MatchesImage);
    waitKey(0);
    /*
    FileStorage fs("FundamentalMat.xml", FileStorage::WRITE);
    fs << "fundamentalMat" <<FundamentalMatrix;
    fs.release();
    */
    
    //利用基础矩阵和相机内参求本质矩阵，SVD分解求相机旋转。
    Mat K_t_2, K_t_3;//相机的内参矩阵。
    Mat F(FundamentalMatrix.size(),K_t_2.type());//数据类型转换。
    FundamentalMatrix.convertTo(F,CV_32FC1);
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
    
    K_t_2 = (Mat_<float>(3,3)<<721.537700, 0.000000, 609.559300, 0.000000,721.537700, 172.854000, 0.000000, 0.000000, 1.000000);
    K_t_3 = (Mat_<float>(3,3)<<721.537700, 0.000000, 609.559300, 0.000000, 721.537700, 172.854000, 0.000000, 0.000000, 1.000000);
    
    Mat EssentialMat;
    EssentialMat =  K_t_2.t()* F* K_t_2;
    
    Mat U, W, Vt, Rotation;
    SVD::compute(EssentialMat, W, U, Vt);
    //Rotation = U * Mat::diag(W) * Vt;//从相机位置t到t+1的旋转矩阵。
    W = (Mat_<float>(3,3)<<0,-1,0, 1,0,0, 0,0,1);//按照计算机视觉中的多视图几何（9.14）公式写的。
    Rotation = U * W * Vt;
    /*
    std::cout<<Mat::diag(W)<<std::endl;
    std::cout<<U<<std::endl;
    std::cout<<Vt<<std::endl;
    */
    std::cout<<Rotation<<std::endl;
    
    return 0;
}
