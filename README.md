# sgm-flow

## descriptions:

These code were written according to the papar [^1] [^2].

performSGM part according to the code [spsstereo](http://ttic.uchicago.edu/~dmcallester/SPS/).

[^1]:K.Yamaguchi, D.McAllester, R.Urtasun: Efﬁcient Joint Segmentation, Occlusion Labeling, Stereo and Flow Estimation. In: ECCV. (2014)
[^2]:K.Yamaguchi, D.Mcallester, R.Urtasun: Robust Monocular Epipolar Flow Estimation.In: CVPR. (2013)
only implemented the SGM flow part, based on left image, without left-right consistence check and SPS smooth.



## environment & dependencies
* opencv 2.4.13
* eigen
* png++
* language: C/C++
* tested MAC 10.12
images and calibration results were get form KITTI2012 flow (image_0).

## details
* change CMakeLists.txt first
* cmake .
* make
* usage: ./flow 000008_10.png 000008_11.png




