#include "image.h"

using namespace cv;

void hstack(cv::Mat& imageLeft, cv::Mat& imageRight, cv::Mat& image)
{
  Size sizeLeft = imageLeft.size();
  Size sizeRight = imageRight.size();
  Size size(sizeLeft.width+sizeRight.width,
            std::max(sizeLeft.height, sizeRight.height));

  image.create(size, imageLeft.type());
  image = Scalar::all(0);

  Mat areaLeft = image(
        Rect(0, 0, sizeLeft.width, sizeLeft.height));
  Mat areaRight = image(
        Rect(sizeLeft.width, 0, sizeRight.width, sizeRight.height));

  imageLeft.copyTo(areaLeft);
  imageRight.copyTo(areaRight);
}
