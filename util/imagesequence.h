#include <stdlib.h>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"

class ImageSequence
{
private:
  std::vector<cv::String> fileList;
  int mSize;
  int currentIndex;
public:
  ImageSequence(const char* path, std::string extension);
  ~ImageSequence();
  void get(cv::Mat& image);
  void increment();
  void decrement();
  int size();
  int getCurrentIndex();
  void setCurrentIndex(int i);
};
