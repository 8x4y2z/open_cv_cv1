#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "opencv2/core/types.hpp"

using namespace cv;
Mat image, dummy;
void cropImage(int action, int x, int y, int flags, void*) {
  static Point tl, br;
  if (action == EVENT_LBUTTONDOWN) {
    tl = {x, y};
  } else if (action == EVENT_LBUTTONUP) {
    br = {x, y};
    rectangle(image, tl, br, Scalar(255, 0, 255), 2);
    imshow("window", image);
    Mat cropped = dummy(Range(tl.y, br.y), Range(tl.x, br.x));
    imwrite("face.png", cropped);
  }
}

int main() {
  image = imread("sample.jpg");
  dummy = image.clone();
  namedWindow("window");
  setMouseCallback("window", cropImage);
  putText(image, "Choose top left corner and drag", Point(10, 30),
          FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
  imshow("window", image);
  waitKey(0);
  return 0;
}
