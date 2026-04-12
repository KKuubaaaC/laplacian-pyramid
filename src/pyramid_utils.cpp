#include "pyramid/pyramid_utils.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace pyramid {

double ComputePSNR(const cv::Mat& original, const cv::Mat& reconstructed, double max_signal) {
    if (original.size() != reconstructed.size()) {
        throw std::invalid_argument(
            "ComputePSNR: size mismatch - original is " + std::to_string(original.cols) + "x" +
            std::to_string(original.rows) + ", reconstructed is " +
            std::to_string(reconstructed.cols) + "x" + std::to_string(reconstructed.rows));
    }
    if (original.type() != CV_32FC1) {
        throw std::invalid_argument(
            "ComputePSNR: original must be CV_32FC1 (single-channel "
            "32-bit float); got OpenCV type id " +
            std::to_string(original.type()));
    }
    if (reconstructed.type() != CV_32FC1) {
        throw std::invalid_argument(
            "ComputePSNR: reconstructed must be CV_32FC1 (single-channel "
            "32-bit float); got OpenCV type id " +
            std::to_string(reconstructed.type()));
    }

    double mse = 0.0;
    for (int y = 0; y < original.rows; ++y) {
        const float* row_o = original.ptr<float>(y);
        const float* row_r = reconstructed.ptr<float>(y);
        for (int x = 0; x < original.cols; ++x) {
            const double diff = static_cast<double>(row_o[x]) - static_cast<double>(row_r[x]);
            mse += diff * diff;
        }
    }
    mse /= static_cast<double>(original.rows * original.cols);
    if (mse == 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    const double peak_sq = max_signal * max_signal;
    return 10.0 * std::log10(peak_sq / mse);
}

}  // namespace pyramid
