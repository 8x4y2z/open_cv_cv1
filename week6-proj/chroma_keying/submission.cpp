#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

#include "opencv2/core.hpp"
#include "opencv2/core/base.hpp"
#include "opencv2/core/hal/interface.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/matx.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"

using namespace std;
using namespace cv;

int sensitivity{0};
int smoothness{0};
int alphaFactor{0};
Mat3b frame;
bool selected{0};
constexpr int maxSensitivity{100};
constexpr int maxSmoothness{10};
constexpr int maxalpha{100};
constexpr double factor{1.85};
constexpr int sensOffset{34};
double bgminH{}, bgmaxH{};

double greenSL{0};
double greenVL{0};
double greenSU{0};
double greenVU{0};
constexpr int kerSize{10};
const char* window{"VideoFrame"};
const char* sensTbar{"tolerance%"};
const char* smoothBar{"smoothness"};
const char* defringeBar{"defringe"};

void bgSensitivityCb(int, void*) {}
void bgColorCb(int action, int x, int y, int flags, void*) {
  static Point tl, br;
  switch (action) {
    case EVENT_LBUTTONDOWN:
      tl = {x, y};
      break;
    case EVENT_LBUTTONUP:
      br = {x, y};
      rectangle(frame, tl, br, Scalar(255, 0, 255), 2);
      imshow(window, frame);
      Mat cropped = frame(Range(tl.y, br.y), Range(tl.x, br.x));
      cvtColor(cropped, cropped, COLOR_BGR2YCrCb);
      Mat channels[3];
      split(cropped, channels);
      Point minloc, maxloc;
      minMaxLoc(channels[0], &bgminH, &bgmaxH, &minloc, &maxloc);
      minMaxLoc(channels[1], &greenSL, &greenSU, &minloc, &maxloc);
      minMaxLoc(channels[2], &greenVL, &greenVU, &minloc, &maxloc);
      selected = 1;
  }
}

void chromaKey(Mat3b& frame, int tolerance, int smoothness) {
  // Convert frame to hsv
  Mat3b hsvimg;
  cvtColor(frame, hsvimg, COLOR_BGR2YCrCb);

  Scalar lower{bgminH, greenSL, greenVL};
  Scalar upper{bgmaxH, greenSU, greenVU};
  Vec3d lowerV{lower[0], lower[1], lower[2]};
  Vec3d upperV{upper[0], upper[1], upper[2]};

  double extraTol = tolerance * sensOffset / 100.0;
  double totalTol = sensOffset + extraTol;
  Size size = frame.size();
  Mat1b mask = Mat1b::zeros(size);
  for (int y{0}; y < size.height; ++y) {
    for (int x{0}; x < size.width; ++x) {
      Vec3d pointV(hsvimg(y, x)[0], hsvimg(y, x)[1], hsvimg(y, x)[2]);
      double dmin = norm(pointV - lowerV);
      double dmax = norm(pointV - upperV);
      if (dmin < totalTol || dmax < totalTol) {
        mask(y, x) = 0;
      } else if (dmin > factor * totalTol && dmax > factor * totalTol) {
        mask(y, x) = 255;

      } else {
        double minDist = min({dmin, dmax});
        double d1 = (factor * totalTol - minDist) / (factor * totalTol);
        uint alpha =
            (uint)(255.0 * d1);  // lower value is close to fg vice-versa
        mask(y,x)=alpha;
      }
    }
  }

  if (smoothness > 0)
  // Perform gausian blur
  {
    smoothness = smoothness % 2 == 1 ? smoothness : smoothness + 1;
    GaussianBlur(mask, mask, Size(smoothness, smoothness), 0);
  }

  const Vec3b bgColorVec(255, 0, 0);
  Mat3b newImage{size};

  for (int y = 0; y < size.height; ++y) {
    for (int x = 0; x < size.width; ++x) {
      uint maskValue = mask(y, x);
      if (maskValue >= 255) {
        newImage(y, x) = frame(y, x);
      } else if (maskValue <= 0) {
        newImage(y, x) = bgColorVec;
      } else {
        double alpha = (1. / (double)(maskValue+alphaFactor*maskValue/100.0));
        newImage(y, x) = (alpha)*frame(y, x) + (1-alpha) * bgColorVec;
      }
    }
  }

  resize(newImage, newImage, Size(800, 600));
  imshow(window, newImage);
}
int main() {
  // VideoCapture cap("greenscreen-asteroid.mp4");
  VideoCapture cap("greenscreen-demo.mp4");

  if (!cap.isOpened()) {
    cout << "Error opening video stream or file\n";
    return -1;
  }
  namedWindow(window, WINDOW_AUTOSIZE);
  setMouseCallback(window, bgColorCb);
  createTrackbar(sensTbar, window, &sensitivity, maxSensitivity,
                 bgSensitivityCb);
  createTrackbar(smoothBar, window, &smoothness, maxSmoothness,
                 bgSensitivityCb);
  createTrackbar(defringeBar, window, &alphaFactor, maxalpha, bgSensitivityCb);
  for (;;) {
    cap >> frame;

    if (frame.empty()) break;

    if (!selected) {
      resize(frame, frame, Size(800, 600));
      imshow(window, frame);
      char c = (char)waitKey(20);
      if (c == 27) break;
    } else {
      chromaKey(frame, sensitivity, smoothness);

      char c = (char)waitKey(1);
      if (c == 27) break;
    }
  }
  cap.release();
  destroyAllWindows();
  return 0;
}
