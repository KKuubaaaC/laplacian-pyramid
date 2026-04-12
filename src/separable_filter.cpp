#include "separable_filter.h"

#include <algorithm>
#include <array>
#include <opencv2/core/mat.hpp>

namespace pyramid {
namespace internal {

int ReflectIndex(int idx, int size) {
    CV_Assert(size > 1);
    // reflect-101: period is 2*(size-1); the border pixel itself is never
    // duplicated. Examples for size=5: -1→1, -2→2, 5→3, 6→2
    const int period = 2 * (size - 1);
    idx %= period;
    if (idx < 0) idx += period;
    if (idx >= size) idx = period - idx;
    return idx;
}

cv::Mat ConvolveRows(const cv::Mat& src, const std::array<float, 5>& kernel) {
    CV_Assert(!src.empty());
    CV_Assert(src.type() == CV_32FC1);

    const int rows = src.rows;
    const int cols = src.cols;
    cv::Mat dst(rows, cols, CV_32FC1);

    for (int y = 0; y < rows; ++y) {
        const float* src_row = src.ptr<float>(y);
        float* dst_row = dst.ptr<float>(y);

        // Left border: x in [0, 2) — uses ReflectIndex for negative offsets
        for (int x = 0; x < std::min(2, cols); ++x) {
            float val = 0.0f;
            for (int k = -2; k <= 2; ++k) {
                val += kernel[static_cast<std::size_t>(k + 2)] * src_row[ReflectIndex(x + k, cols)];
            }
            dst_row[x] = val;
        }

        // Inner loop: x in [2, cols-2) — no branches, direct pointer arithmetic
        for (int x = 2; x < cols - 2; ++x) {
            dst_row[x] = kernel[0] * src_row[x - 2] + kernel[1] * src_row[x - 1] +
                         kernel[2] * src_row[x] + kernel[3] * src_row[x + 1] +
                         kernel[4] * src_row[x + 2];
        }

        // Right border: x in [max(2, cols-2), cols) — uses ReflectIndex for
        // out-of-bounds positive offsets
        for (int x = std::max(2, cols - 2); x < cols; ++x) {
            float val = 0.0f;
            for (int k = -2; k <= 2; ++k) {
                val += kernel[static_cast<std::size_t>(k + 2)] * src_row[ReflectIndex(x + k, cols)];
            }
            dst_row[x] = val;
        }
    }

    return dst;
}

cv::Mat ConvolveCols(const cv::Mat& src, const std::array<float, 5>& kernel) {
    CV_Assert(!src.empty());
    CV_Assert(src.type() == CV_32FC1);

    const int rows = src.rows;
    const int cols = src.cols;
    cv::Mat dst(rows, cols, CV_32FC1);

    // Top border: y in [0, 2) — uses ReflectIndex for negative row offsets
    for (int y = 0; y < std::min(2, rows); ++y) {
        float* dst_row = dst.ptr<float>(y);
        for (int x = 0; x < cols; ++x) {
            float val = 0.0f;
            for (int k = -2; k <= 2; ++k) {
                val += kernel[static_cast<std::size_t>(k + 2)] *
                       src.ptr<float>(ReflectIndex(y + k, rows))[x];
            }
            dst_row[x] = val;
        }
    }

    // Inner loop: y in [2, rows-2) — prefetch 5 row pointers, then branchless
    // column sweep. No bounds checks needed for any of the 5 offsets
    for (int y = 2; y < rows - 2; ++y) {
        const float* r0 = src.ptr<float>(y - 2);
        const float* r1 = src.ptr<float>(y - 1);
        const float* r2 = src.ptr<float>(y);
        const float* r3 = src.ptr<float>(y + 1);
        const float* r4 = src.ptr<float>(y + 2);
        float* dst_row = dst.ptr<float>(y);
        for (int x = 0; x < cols; ++x) {
            dst_row[x] = kernel[0] * r0[x] + kernel[1] * r1[x] + kernel[2] * r2[x] +
                         kernel[3] * r3[x] + kernel[4] * r4[x];
        }
    }

    // Bottom border: y in [max(2, rows-2), rows) — uses ReflectIndex for
    // out-of-bounds positive row offsets
    for (int y = std::max(2, rows - 2); y < rows; ++y) {
        float* dst_row = dst.ptr<float>(y);
        for (int x = 0; x < cols; ++x) {
            float val = 0.0f;
            for (int k = -2; k <= 2; ++k) {
                val += kernel[static_cast<std::size_t>(k + 2)] *
                       src.ptr<float>(ReflectIndex(y + k, rows))[x];
            }
            dst_row[x] = val;
        }
    }

    return dst;
}

cv::Mat GaussianBlur(const cv::Mat& input, const std::array<float, 5>& kernel) {
    CV_Assert(!input.empty());
    CV_Assert(input.type() == CV_32FC1);

    // Separable: horizontal pass first, then vertical
    return ConvolveCols(ConvolveRows(input, kernel), kernel);
}

}  // namespace internal
}  // namespace pyramid
