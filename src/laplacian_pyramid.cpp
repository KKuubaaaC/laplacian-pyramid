#include "pyramid/laplacian_pyramid.h"

#include <cassert>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <stdexcept>
#include <vector>

#include "image_arithmetic.h"
#include "pyramid_ops.h"
#include "separable_filter.h"

namespace {

constexpr int kMaxPyramidLevels = 8;

}  // namespace

namespace pyramid {

namespace internal {

// Reduce / Expand — building blocks of the Laplacian pyramid

cv::Mat Reduce(const cv::Mat& img) {
    CV_Assert(!img.empty());
    CV_Assert(img.type() == CV_32FC1);
    return Downsample(GaussianBlur(img));
}

cv::Mat Expand(const cv::Mat& img, int target_rows, int target_cols) {
    CV_Assert(!img.empty());
    CV_Assert(img.type() == CV_32FC1);
    CV_Assert(target_rows > 0 && target_cols > 0);

    // Upsample places source pixels at even positions; GaussianBlur with
    // kExpandKernel (2× taps per axis, sum 2 each pass) restores ×4 amplitude
    return GaussianBlur(Upsample(img, target_rows, target_cols), kExpandKernel);
}

}  // namespace internal

// LaplacianPyramid implementation

LaplacianPyramid::LaplacianPyramid(std::vector<cv::Mat> laplacian_levels, cv::Mat gaussian_top,
                                   PyramidParams params)
    : laplacian_levels_(std::move(laplacian_levels)),
      gaussian_top_(std::move(gaussian_top)),
      params_(params) {}

std::optional<LaplacianPyramid> LaplacianPyramid::Build(const cv::Mat& input,
                                                        PyramidParams params) {
    if (params.num_levels <= 0 || params.num_levels > kMaxPyramidLevels) {
        std::cerr << "Error: num_levels must be in [1, " << kMaxPyramidLevels << "], got "
                  << params.num_levels << "\n";
        return std::nullopt;
    }
    if (input.empty()) {
        std::cerr << "Error: input image is empty\n";
        return std::nullopt;
    }
    if (input.type() != CV_32FC1) {
        std::cerr << "Error: expected CV_32FC1, got type " << input.type() << "\n";
        return std::nullopt;
    }
    // Each level halves dimensions; require at least 2×2 at the top level
    const int min_dim = std::min(input.rows, input.cols);
    if (min_dim < (1 << params.num_levels)) {
        std::cerr << "Error: image too small for " << params.num_levels << " levels\n";
        return std::nullopt;
    }

    const int n = params.num_levels;

    // Build Gaussian pyramid: gaussian[0] = input, gaussian[i+1] = Reduce(gaussian[i])
    std::vector<cv::Mat> gaussian(static_cast<std::size_t>(n + 1));
    gaussian[0] = input;
    for (int i = 0; i < n; ++i) {
        gaussian[static_cast<std::size_t>(i + 1)] =
            internal::Reduce(gaussian[static_cast<std::size_t>(i)]);
    }

    // Compute Laplacian levels: laplacian[i] = gaussian[i] - Expand(gaussian[i+1])
    std::vector<cv::Mat> laplacian_levels(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        const cv::Mat& g_fine = gaussian[static_cast<std::size_t>(i)];
        const cv::Mat& g_coarse = gaussian[static_cast<std::size_t>(i + 1)];
        const cv::Mat expanded = internal::Expand(g_coarse, g_fine.rows, g_fine.cols);
        laplacian_levels[static_cast<std::size_t>(i)] = internal::Subtract(g_fine, expanded);
    }

    cv::Mat gaussian_top = std::move(gaussian[static_cast<std::size_t>(n)]);

    return LaplacianPyramid(std::move(laplacian_levels), std::move(gaussian_top), params);
}

cv::Mat LaplacianPyramid::Reconstruct() const {
    // Start from the coarsest Gaussian level and expand upwards
    // At each step: g[i] = Expand(g[i+1]) + L[i]
    // Because L[i] = g[i] - Expand(g[i+1]) was stored in Build, and Expand
    // is deterministic, the round-trip is algebraically perfect
    const int n = NumLevels();
    CV_Assert(n > 0);
    const cv::Mat* coarse = &gaussian_top_;
    cv::Mat current;
    for (int i = n - 1; i >= 0; --i) {
        const cv::Mat& lap = laplacian_levels_[static_cast<std::size_t>(i)];
        current = internal::Add(internal::Expand(*coarse, lap.rows, lap.cols), lap);
        coarse = &current;
    }
    return current;
}

int LaplacianPyramid::NumLevels() const noexcept {
    return static_cast<int>(laplacian_levels_.size());
}

const cv::Mat& LaplacianPyramid::LaplacianLevel(int level) const {
    if (level < 0 || level >= NumLevels()) {
        throw std::out_of_range("LaplacianLevel: level out of range");
    }
    return laplacian_levels_[static_cast<std::size_t>(level)];
}

const cv::Mat& LaplacianPyramid::GaussianTop() const noexcept { return gaussian_top_; }

}  // namespace pyramid
