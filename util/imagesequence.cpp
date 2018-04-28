#include "imagesequence.h"

ImageSequence::ImageSequence(const char* path, std::string extension)
{
  std::vector<cv::String> list;
  cv::String pattern = path;
  glob(pattern, list);

  for (int i = 0; i < list.size(); i++)
  {
    cv::String s = list.at(i);
    cv::String ext = cv::String(extension);
    if (!s.substr(s.size() - 3, 3).compare(ext))
    {
      fileList.push_back(s);
    }
  }
  currentIndex = 0;
  mSize = fileList.size();
}

ImageSequence::~ImageSequence()
{

}

void ImageSequence::get(cv::Mat& image)
{
  image = cv::imread(fileList[currentIndex], cv::IMREAD_GRAYSCALE);
}

void ImageSequence::increment()
{
  currentIndex++;
  if (currentIndex >= mSize)
  {
    currentIndex = 0;
  }
}

void ImageSequence::decrement()
{
  currentIndex--;
  if (currentIndex < 0)
  {
    currentIndex = mSize - 1;
  }
}

int ImageSequence::size()
{
  return mSize;
}

int ImageSequence::getCurrentIndex()
{
  return currentIndex;
}

void ImageSequence::setCurrentIndex(int i)
{
    currentIndex = i;
}
