#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/core/types.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace std;

Mat convertBGRtoHSV(Mat image){
    ///
    /// YOUR CODE HERE
    ///

    Mat out;
    Mat hsv[3];
    Mat tmp{image.clone()};
    tmp.convertTo(tmp,CV_32F);
    tmp=tmp/255.0;
    Mat bgr[3];
    split(tmp,bgr);
    for(int i{0};i<3;++i) hsv[i]=Mat(tmp.size(),CV_32F,Scalar(0,0,0));

    int height,width;
    float h,s,v,mv,r,g,b;
    height=tmp.size().height;
    width=tmp.size().width;
    for(int i{0};i<height;++i){
        for(int j{0};j<width;++j){
            r=bgr[2].at<float>(i,j);
            g=bgr[1].at<float>(i,j);
            b=bgr[0].at<float>(i,j);
            v=max({b,g,r});
            mv=min({b,g,r});
            if(v==0) s=0;
            else s=(v-mv)/v;
            if(v==b) h=240+60*(r-g)/(v-mv);
            else if (v==g) h=120+60*(b-r)/(v-mv);
            else if (v==r) h=60*(g-b)/(v-mv);
            if (h<0) h=h+360;
            v=255*v;
            s=255*s;
            h=h/2;
            hsv[0].at<float>(i,j)=h;
            hsv[1].at<float>(i,j)=s;
            hsv[2].at<float>(i,j)=v;
        }
    }
    cout<<"done\n";
    merge(hsv, 3, out);
    out.convertTo(out,CV_8U);
    return out;
}

int main(){
  Mat img,out;
  img=imread("sample.jpg");
  out=convertBGRtoHSV(img);
  imshow("Out",out);
  waitKey(0);
  destroyAllWindows();
  return 0;

}
