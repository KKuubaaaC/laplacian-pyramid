#ifndef SRC_SEPARABLE_FILTER_H_
#define SRC_SEPARABLE_FILTER_H_

// INTERNAL HEADER — not part of the public API. Do not include from user code

#include <array>
#include <opencv2/core/mat.hpp>

namespace pyramid {
namespace internal {

static constexpr std::array<float, 5> kBurtAdelsonKernel = {0.05f, 0.25f, 0.4f, 0.25f, 0.05f};

// Twice the Burt–Adelson taps. Used after upsampling: two separable passes each
// scale by sum(kernel)=2, for a net ×4 gain that compensates upsample energy loss
static constexpr std::array<float, 5> kExpandKernel = {
    kBurtAdelsonKernel[0] * 2.0f, kBurtAdelsonKernel[1] * 2.0f, kBurtAdelsonKernel[2] * 2.0f,
    kBurtAdelsonKernel[3] * 2.0f, kBurtAdelsonKernel[4] * 2.0f};

// Reflects an index at image borders using reflect-101 (border pixel not
// duplicated), matching OpenCV BORDER_REFLECT_101.
// Example for size=5: ..., -2→2, -1→1, 0→0, 4→4, 5→3, 6→2, ...
[[nodiscard]] int ReflectIndex(int idx, int size);

// Applies a separable 1D kernel along each row.
// kernel is applied with radius 2 (kernel size 5)
[[nodiscard]] cv::Mat ConvolveRows(const cv::Mat& src, const std::array<float, 5>& kernel);

// Applies a separable 1D kernel along each column
[[nodiscard]] cv::Mat ConvolveCols(const cv::Mat& src, const std::array<float, 5>& kernel);

// Applies a full separable Gaussian blur using the Burt-Adelson kernel.
// Equivalent to ConvolveCols(ConvolveRows(input, kernel), kernel)
[[nodiscard]] cv::Mat GaussianBlur(const cv::Mat& input,
                                   const std::array<float, 5>& kernel = kBurtAdelsonKernel);

}  // namespace internal
}  // namespace pyramid

#endif  // SRC_SEPARABLE_FILTER_H_
