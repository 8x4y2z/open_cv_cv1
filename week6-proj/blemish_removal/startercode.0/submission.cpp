#include <cstddef>
#include <cstdio>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "opencv2/core.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/photo.hpp"

using namespace cv;
using namespace std;
constexpr int PATCH_RADIUS{15};
const char* WINDOW_NAME{"Blemish"};

Mat img;
Mat imgClone;

float calcAvgSobel(Mat& image) {
  Mat sobelx, sobely;
  // Apply sobel filter with only x gradient
  Sobel(image, sobelx, CV_32F, 1, 0);

  // Apply sobel filter with only y gradient
  Sobel(image, sobely, CV_32F, 0, 1);
  // Perform sqrt(x^2,y^2)
  multiply(sobelx, sobelx, sobelx);
  multiply(sobely, sobely, sobely);
  Mat grads = sobelx + sobely;
  sqrt(grads, grads);
  Scalar s = mean(grads);
  return s[0];
}

void getPatch(Mat& image, Mat& out, int h1, int h2, int v1, int v2) {
  if (v1 <= v2 && h1 <= h2)
    out = image(Range(v1, v2), Range(h1, h2));
  else if (v1 > v2 && h1 <= h2)
    out = image(Range(v2, v1), Range(h1, h2));
  else if (v1 <= v2 && h1 > h2)
    out = image(Range(v1, v2), Range(h2, h1));
  else
    out = image(Range(v2, v1), Range(h2, h1));
}
void onMouse(int event, int x, int y, int flags, void*) {
  if (event == EVENT_LBUTTONDOWN) {
    Mat patch = img(Range(y - PATCH_RADIUS, y + PATCH_RADIUS),
                    Range(x - PATCH_RADIUS, x + PATCH_RADIUS));

    cvtColor(patch, patch, COLOR_BGR2GRAY);
    float grad = calcAvgSobel(patch);
    int h1, v1, h2, v2;
    // get neighbouting areas
    Mat patches[4];
    for (int i{0};i<4;++i){
      switch (i) {
        case 0:
          h1=x - 3 * PATCH_RADIUS;
          h2=x-PATCH_RADIUS;
          v1= y - PATCH_RADIUS;
          v2=y + PATCH_RADIUS;
          break;
        case 1:
          h1=x + 3 * PATCH_RADIUS;
          h2=x+PATCH_RADIUS;
          v1= y - PATCH_RADIUS;
          v2=y + PATCH_RADIUS;
          break;
        case 2:
          h1=x -   PATCH_RADIUS;
          h2=x+PATCH_RADIUS;
          v1= y -3* PATCH_RADIUS;
          v2=y -PATCH_RADIUS;
          break;
        case 3:
          h1=x -  PATCH_RADIUS;
          h2=x+PATCH_RADIUS;
          v1= y +3* PATCH_RADIUS;
          v2=y + PATCH_RADIUS;
          break;
      }
      h1=max({0,h1});
      h1=min({h1,img.size().width});
      h2=max({0,h2});
      h2=min({h2,img.size().width});

      v1=max({0,v1});
      v1=min({v1,img.size().height});
      v2=max({0,v2});
      v2=min({v2,img.size().height});
      getPatch(img, patches[i],h1,h2,v1,v2);
}
    float grads[4];
    float neighG;
    int min;
    Mat patchClone;
    float minGrad{grad};
    // get neighbout with lowest grad
    for (int i{0}; i < 4; ++i) {
      patchClone = patches[i].clone();
      cvtColor(patchClone, patchClone, COLOR_BGR2GRAY);
      neighG = calcAvgSobel(patchClone);
      if (neighG <= minGrad) {
        min = i;
        minGrad = neighG;
      };
    }
    Mat selected{patches[min]};
    int roiHeight = selected.size().height;
    int roiWidth = selected.size().width;
    Mat mask{255 * Mat::ones(selected.rows, selected.cols, selected.depth())};
    Point center(x, y);
    Mat output;
    seamlessClone(selected, imgClone, mask, center, output, NORMAL_CLONE);
    imshow(WINDOW_NAME, output);
    imgClone=output;
  } else if (event == EVENT_RBUTTONDOWN) {
    imgClone=img.clone();
    imshow(WINDOW_NAME, img);
  }
}

int main() {
  img = imread("blemish.png");
  imgClone=img.clone();
  namedWindow(WINDOW_NAME);
  setMouseCallback(WINDOW_NAME, onMouse, NULL);
  imshow(WINDOW_NAME, img);
  while (true) {
    int c;
    // imshow(WINDOW_NAME, img);
    c = waitKey(20);
    if ((char)c == 27) break;
  }
  destroyAllWindows();
  return 0;
}
