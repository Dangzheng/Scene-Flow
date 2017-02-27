# sgm-flow

## descriptions:

These code were written according to the papar.

```
K.Yamaguchi, D.McAllester, R.Urtasun: Efﬁcient Joint Segmentation, Occlusion Labeling, Stereo and Flow Estimation. In: ECCV. (2014)
```
```
K.Yamaguchi, D.Mcallester, R.Urtasun: Robust Monocular Epipolar Flow Estimation.In: CVPR. (2013)

```
performSGM function was according to the code [spsstereo](http://ttic.uchicago.edu/~dmcallester/SPS/). 

I only implemented the **SGM flow part**, based on the left image, without left-right consistence check and SPS smooth.

## environment & dependencies
* opencv 2.4.13
* eigen
* png++
* language: C/C++, matlab(only some tools)
* tested MAC 10.12

images and calibration results were get form KITTI 2012 flow (image_0).

## details
* change CMakeLists.txt first
* cmake .
* make
* usage: ./flow 000008_10.png 000008_11.png

matlab tools contains parameters estimation part, C++ code calculate rotation vector by SVD.

* input image(left)
![left_img](https://github.com/Dangzheng/Scene-Flow/raw/master/000008_10.png)

* output image
![out](https://github.com/Dangzheng/Scene-Flow/raw/master/out.png)

* terminal display (if all settings are correct, terminal output will looks like this.)
![display](https://github.com/Dangzheng/Scene-Flow/raw/master/display.png)

