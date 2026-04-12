#include "image_arithmetic.h"

#include <gtest/gtest.h>

#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>

namespace pyramid {
namespace internal {

// ---------------------------------------------------------------------------
// Subtract
// ---------------------------------------------------------------------------

TEST(ImageArithmeticTest, SubtractSelfIsZero) {
    cv::RNG rng(42);
    cv::Mat a(64, 64, CV_32FC1);
    rng.fill(a, cv::RNG::UNIFORM, 0.0f, 1.0f);

    const cv::Mat result = Subtract(a, a);

    ASSERT_EQ(result.size(), a.size());
    ASSERT_EQ(result.type(), CV_32FC1);
    for (int y = 0; y < result.rows; ++y) {
        const float* row = result.ptr<float>(y);
        for (int x = 0; x < result.cols; ++x) {
            EXPECT_FLOAT_EQ(row[x], 0.0f) << "at (" << y << "," << x << ")";
        }
    }
}

TEST(ImageArithmeticTest, SubtractPreservesNegativeValues) {
    cv::Mat a(4, 4, CV_32FC1, cv::Scalar(100.0f));
    cv::Mat b(4, 4, CV_32FC1, cv::Scalar(120.0f));

    const cv::Mat result = Subtract(a, b);

    ASSERT_EQ(result.size(), a.size());
    for (int y = 0; y < result.rows; ++y) {
        const float* row = result.ptr<float>(y);
        for (int x = 0; x < result.cols; ++x) {
            EXPECT_FLOAT_EQ(row[x], -20.0f) << "at (" << y << "," << x << ")";
        }
    }
}

// ---------------------------------------------------------------------------
// Add
// ---------------------------------------------------------------------------

TEST(ImageArithmeticTest, AddZeroIsIdentity) {
    cv::RNG rng(42);
    cv::Mat a(64, 64, CV_32FC1);
    rng.fill(a, cv::RNG::UNIFORM, -1.0f, 1.0f);
    const cv::Mat zeros = cv::Mat::zeros(64, 64, CV_32FC1);

    const cv::Mat result = Add(a, zeros);

    ASSERT_EQ(result.size(), a.size());
    for (int y = 0; y < result.rows; ++y) {
        const float* row_r = result.ptr<float>(y);
        const float* row_a = a.ptr<float>(y);
        for (int x = 0; x < result.cols; ++x) {
            EXPECT_FLOAT_EQ(row_r[x], row_a[x]) << "at (" << y << "," << x << ")";
        }
    }
}

TEST(ImageArithmeticTest, AddIsCommutative) {
    cv::RNG rng(42);
    cv::Mat a(32, 32, CV_32FC1);
    cv::Mat b(32, 32, CV_32FC1);
    rng.fill(a, cv::RNG::UNIFORM, 0.0f, 1.0f);
    rng.fill(b, cv::RNG::UNIFORM, 0.0f, 1.0f);

    const cv::Mat ab = Add(a, b);
    const cv::Mat ba = Add(b, a);

    ASSERT_EQ(ab.size(), ba.size());
    for (int y = 0; y < ab.rows; ++y) {
        const float* row_ab = ab.ptr<float>(y);
        const float* row_ba = ba.ptr<float>(y);
        for (int x = 0; x < ab.cols; ++x) {
            EXPECT_FLOAT_EQ(row_ab[x], row_ba[x]) << "at (" << y << "," << x << ")";
        }
    }
}

// ---------------------------------------------------------------------------
// Downsample
// ---------------------------------------------------------------------------

TEST(ImageArithmeticTest, DownsampleHalvesDimensions) {
    const cv::Mat img(256, 256, CV_32FC1, cv::Scalar(1.0f));
    const cv::Mat result = Downsample(img);

    EXPECT_EQ(result.rows, 128);
    EXPECT_EQ(result.cols, 128);
    EXPECT_EQ(result.type(), CV_32FC1);
}

// ---------------------------------------------------------------------------
// Upsample
// ---------------------------------------------------------------------------

TEST(ImageArithmeticTest, UpsampleDoublesDimensions) {
    const cv::Mat img(128, 128, CV_32FC1, cv::Scalar(1.0f));
    const cv::Mat result = Upsample(img, 256, 256);

    EXPECT_EQ(result.rows, 256);
    EXPECT_EQ(result.cols, 256);
    EXPECT_EQ(result.type(), CV_32FC1);
}

// ---------------------------------------------------------------------------
// Roundtrip
// ---------------------------------------------------------------------------

TEST(ImageArithmeticTest, DownsampleAfterUpsamplePreservesEvenPositions) {
    // Build a known source image.
    cv::RNG rng(42);
    cv::Mat src(64, 64, CV_32FC1);
    rng.fill(src, cv::RNG::UNIFORM, 0.0f, 1.0f);

    const cv::Mat upsampled = Upsample(src, 128, 128);
    const cv::Mat roundtrip = Downsample(upsampled);

    ASSERT_EQ(roundtrip.size(), src.size());
    for (int y = 0; y < src.rows; ++y) {
        const float* row_src = src.ptr<float>(y);
        const float* row_rt = roundtrip.ptr<float>(y);
        for (int x = 0; x < src.cols; ++x) {
            EXPECT_FLOAT_EQ(row_rt[x], row_src[x])
                << "roundtrip mismatch at (" << y << "," << x << ")";
        }
    }
}

}  // namespace internal
}  // namespace pyramid
