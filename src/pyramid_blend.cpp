#include "pyramid/pyramid_blend.h"

#include <opencv2/core/mat.hpp>
#include <vector>

#include "image_arithmetic.h"
#include "pyramid/laplacian_pyramid.h"
#include "pyramid_ops.h"

namespace pyramid {

namespace {

[[nodiscard]] cv::Mat BlendAtLevel(const cv::Mat& la, const cv::Mat& lb, const cv::Mat& gm) {
    CV_Assert(!la.empty());
    CV_Assert(la.size() == lb.size());
    CV_Assert(la.size() == gm.size());
    CV_Assert(la.type() == CV_32FC1 && lb.type() == CV_32FC1 && gm.type() == CV_32FC1);

    cv::Mat out(la.rows, la.cols, CV_32FC1);
    for (int y = 0; y < la.rows; ++y) {
        const float* row_la = la.ptr<float>(y);
        const float* row_lb = lb.ptr<float>(y);
        const float* row_gm = gm.ptr<float>(y);
        float* row_out = out.ptr<float>(y);
        for (int x = 0; x < la.cols; ++x) {
            const float g = row_gm[x];
            row_out[x] = g * row_la[x] + (1.0f - g) * row_lb[x];
        }
    }
    return out;
}

[[nodiscard]] cv::Mat CollapseBlendedPyramid(const std::vector<cv::Mat>& laplacian_levels,
                                             const cv::Mat& gaussian_top) {
    CV_Assert(!laplacian_levels.empty());
    const int n = static_cast<int>(laplacian_levels.size());
    const cv::Mat* coarse = &gaussian_top;
    cv::Mat current;
    for (int i = n - 1; i >= 0; --i) {
        const cv::Mat& lap = laplacian_levels[static_cast<std::size_t>(i)];
        current = internal::Add(internal::Expand(*coarse, lap.rows, lap.cols), lap);
        coarse = &current;
    }
    return current;
}

[[nodiscard]] std::vector<cv::Mat> BuildGaussianPyramid(const cv::Mat& base, int num_levels) {
    std::vector<cv::Mat> g(static_cast<std::size_t>(num_levels + 1));
    g[0] = base;
    for (int i = 0; i < num_levels; ++i) {
        const std::size_t iu = static_cast<std::size_t>(i);
        g[iu + 1U] = internal::Reduce(g[iu]);
    }
    return g;
}

}  // namespace

std::optional<cv::Mat> BlendLaplacianPyramids(const cv::Mat& image_a, const cv::Mat& image_b,
                                              const cv::Mat& mask, PyramidParams params) {
    if (image_a.empty() || image_b.empty() || mask.empty()) {
        return std::nullopt;
    }
    if (image_a.type() != CV_32FC1 || image_b.type() != CV_32FC1 || mask.type() != CV_32FC1) {
        return std::nullopt;
    }
    if (image_a.size() != image_b.size() || image_a.size() != mask.size()) {
        return std::nullopt;
    }
    if (params.num_levels <= 0) {
        return std::nullopt;
    }

    const auto pyr_a = LaplacianPyramid::Build(image_a, params);
    const auto pyr_b = LaplacianPyramid::Build(image_b, params);
    if (!pyr_a.has_value() || !pyr_b.has_value()) {
        return std::nullopt;
    }

    const int n = params.num_levels;
    const std::vector<cv::Mat> mask_pyr = BuildGaussianPyramid(mask, n);

    std::vector<cv::Mat> blended_lap(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        blended_lap[static_cast<std::size_t>(i)] =
            BlendAtLevel(pyr_a->LaplacianLevel(i), pyr_b->LaplacianLevel(i),
                         mask_pyr[static_cast<std::size_t>(i)]);
    }

    const cv::Mat blended_top = BlendAtLevel(pyr_a->GaussianTop(), pyr_b->GaussianTop(),
                                             mask_pyr[static_cast<std::size_t>(n)]);

    return CollapseBlendedPyramid(blended_lap, blended_top);
}

}  // namespace pyramid
