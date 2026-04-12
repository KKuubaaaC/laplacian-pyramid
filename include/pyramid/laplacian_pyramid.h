#ifndef PYRAMID_LAPLACIAN_PYRAMID_H_
#define PYRAMID_LAPLACIAN_PYRAMID_H_

#include <opencv2/core/mat.hpp>
#include <optional>
#include <vector>

#include "pyramid/pyramid_types.h"

namespace pyramid {

// Laplacian pyramid decomposition and perfect reconstruction
//
// Build the pyramid with LaplacianPyramid::Build(), then reconstruct the
// original image (to floating-point precision) with Reconstruct()
//
// The class is move-only: copying a pyramid would duplicate potentially large
// image data, so copy operations are explicitly deleted
class LaplacianPyramid {
   public:
    LaplacianPyramid(LaplacianPyramid&&) noexcept = default;
    LaplacianPyramid& operator=(LaplacianPyramid&&) noexcept = default;
    LaplacianPyramid(const LaplacianPyramid&) = delete;
    LaplacianPyramid& operator=(const LaplacianPyramid&) = delete;

    // Builds the pyramid from a CV_32FC1 image
    // Returns nullopt when input is empty, not CV_32FC1, or too small for the
    // requested number of levels
    [[nodiscard]] static std::optional<LaplacianPyramid> Build(const cv::Mat& input,
                                                               PyramidParams params = {});

    // Reconstructs the original image from the pyramid
    // With CV_32F the error is at float32 ULP scale (typically ~150–190 dB PSNR
    // vs. original for random imagery, depending on depth and content)
    [[nodiscard]] cv::Mat Reconstruct() const;

    // Number of Laplacian levels (does NOT include the Gaussian top)
    [[nodiscard]] int NumLevels() const noexcept;

    // Returns the Laplacian level at the given index (0 == finest)
    // Throws std::out_of_range if level is outside [0, NumLevels())
    [[nodiscard]] const cv::Mat& LaplacianLevel(int level) const;

    // Returns the coarsest Gaussian level stored at the top of the pyramid.
    [[nodiscard]] const cv::Mat& GaussianTop() const noexcept;

   private:
    explicit LaplacianPyramid(std::vector<cv::Mat> laplacian_levels, cv::Mat gaussian_top,
                              PyramidParams params);

    std::vector<cv::Mat> laplacian_levels_;
    cv::Mat gaussian_top_;
    PyramidParams params_;
};

}  // namespace pyramid

#endif  // PYRAMID_LAPLACIAN_PYRAMID_H_
