# Scene-Flow
cook a dynamic scene flow. 
## instruction
这部分代码是实现flow部分的代码
0.png 1.png是t时刻和t+1时刻的left view图像，3.png是t时刻Stereo相机的right view。

## 流程
### 2016.8.14
 1、利用SIFT计算基础矩阵
 2、求出的本质矩阵转为本质矩阵
 3、根据本质矩阵求取旋转矩阵