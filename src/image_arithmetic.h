#ifndef SRC_IMAGE_ARITHMETIC_H_
#define SRC_IMAGE_ARITHMETIC_H_

// INTERNAL HEADER — not part of the public API. Do not include from user code

#include <opencv2/core/mat.hpp>

namespace pyramid {
namespace internal {

// Returns dst = a + b (element-wise, CV_32FC1)
[[nodiscard]] cv::Mat Add(const cv::Mat& a, const cv::Mat& b);

// Returns dst = a - b (element-wise, CV_32FC1, may produce negative values)
[[nodiscard]] cv::Mat Subtract(const cv::Mat& a, const cv::Mat& b);

// Returns a half-resolution image by keeping every other pixel (no filtering)
// Output size: (rows/2) x (cols/2)
[[nodiscard]] cv::Mat Downsample(const cv::Mat& img);

// Returns a double-resolution image by inserting zeros between samples
// Output size: target_rows x target_cols
[[nodiscard]] cv::Mat Upsample(const cv::Mat& img, int target_rows, int target_cols);

}  // namespace internal
}  // namespace pyramid

#endif  // SRC_IMAGE_ARITHMETIC_H_
