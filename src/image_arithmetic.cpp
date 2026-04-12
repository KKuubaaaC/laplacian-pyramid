#include "image_arithmetic.h"

#include <opencv2/core/mat.hpp>

namespace pyramid {
namespace internal {

cv::Mat Add(const cv::Mat& a, const cv::Mat& b) {
    CV_Assert(!a.empty());
    CV_Assert(a.size() == b.size());
    CV_Assert(a.type() == CV_32FC1);
    CV_Assert(b.type() == CV_32FC1);

    cv::Mat result(a.rows, a.cols, CV_32FC1);
    for (int y = 0; y < a.rows; ++y) {
        const float* row_a = a.ptr<float>(y);
        const float* row_b = b.ptr<float>(y);
        float* row_out = result.ptr<float>(y);
        for (int x = 0; x < a.cols; ++x) {
            row_out[x] = row_a[x] + row_b[x];
        }
    }
    return result;
}

cv::Mat Subtract(const cv::Mat& a, const cv::Mat& b) {
    CV_Assert(!a.empty());
    CV_Assert(a.size() == b.size());
    CV_Assert(a.type() == CV_32FC1);
    CV_Assert(b.type() == CV_32FC1);

    cv::Mat result(a.rows, a.cols, CV_32FC1);
    for (int y = 0; y < a.rows; ++y) {
        const float* row_a = a.ptr<float>(y);
        const float* row_b = b.ptr<float>(y);
        float* row_out = result.ptr<float>(y);
        for (int x = 0; x < a.cols; ++x) {
            row_out[x] = row_a[x] - row_b[x];
        }
    }
    return result;
}

cv::Mat Downsample(const cv::Mat& img) {
    CV_Assert(!img.empty());
    CV_Assert(img.type() == CV_32FC1);

    // Keep every even-indexed row and column (subsample by 2)
    const int new_rows = img.rows / 2;
    const int new_cols = img.cols / 2;
    cv::Mat result(new_rows, new_cols, CV_32FC1);
    for (int y = 0; y < new_rows; ++y) {
        const float* src_row = img.ptr<float>(2 * y);
        float* dst_row = result.ptr<float>(y);
        for (int x = 0; x < new_cols; ++x) {
            dst_row[x] = src_row[2 * x];
        }
    }
    return result;
}

cv::Mat Upsample(const cv::Mat& img, int target_rows, int target_cols) {
    CV_Assert(!img.empty());
    CV_Assert(img.type() == CV_32FC1);
    CV_Assert(target_rows > 0 && target_cols > 0);

    // Place source pixels at even positions; odd positions remain zero
    // This is the inverse of Downsample: Downsample(Upsample(x)) == x
    cv::Mat result = cv::Mat::zeros(target_rows, target_cols, CV_32FC1);
    for (int y = 0; y < img.rows; ++y) {
        const float* src_row = img.ptr<float>(y);
        float* dst_row = result.ptr<float>(2 * y);
        for (int x = 0; x < img.cols; ++x) {
            dst_row[2 * x] = src_row[x];
        }
    }
    return result;
}

}  // namespace internal
}  // namespace pyramid
