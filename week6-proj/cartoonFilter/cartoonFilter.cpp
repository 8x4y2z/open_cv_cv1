#include <opencv2/opencv.hpp>
#include <string>

#include "opencv2/core.hpp"
#include "opencv2/core/hal/interface.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;
using namespace std;

Mat cartoonFilter(Mat image, int arguments = 0) {
  Mat pencilSketchImage;
  Mat laplacian;
  int kernelSize = 3;

  // Blur
  GaussianBlur(image, laplacian, Size(3, 3), 0, 0);
  Mat imgClone{laplacian.clone()};
  // Convert to graysclae
  cvtColor(laplacian, laplacian, COLOR_BGR2GRAY);
  // Laplacian filter
  Laplacian(laplacian, pencilSketchImage, CV_32F, kernelSize, 1, 0);
  // Convert back to 8U
  pencilSketchImage.convertTo(pencilSketchImage, CV_8U);
  // binary threshold
  threshold(pencilSketchImage, pencilSketchImage, 10, 255, THRESH_BINARY_INV);

  bitwise_and(imgClone,imgClone,laplacian,pencilSketchImage);

  return laplacian;
}

int main() {
  string imagePath = "trump.jpg";
  Mat image = imread(imagePath);
  Mat pencilSketchImage = cartoonFilter(image);
  imshow("Sketch", pencilSketchImage);
  waitKey(0);
  destroyAllWindows();
  return 0;
}
