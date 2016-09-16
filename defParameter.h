// 参数设置暂时维持和Stereo一致，在程序完成后调试过程中再进行微调。
// The number of superpixel
int superpixelTotal = 1000;
// The number of iterations
int outerIterationTotal = 10;
int innerIterationTotal = 10;

// Weight parameters
double lambda_pos = 500.0;
double lambda_depth = 2000.0;
double lambda_bou = 1000.0;
double lambda_smo = 400.0;

// Inlier threshold
double lambda_d = 3.0;

// Penalty values
double lambda_hinge = 5.0;
double lambda_occ = 15.0;
double lambda_pen = 30.0; // penalty impossible.