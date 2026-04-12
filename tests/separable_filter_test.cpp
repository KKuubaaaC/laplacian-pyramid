#include "separable_filter.h"

#include <gtest/gtest.h>

#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>

namespace pyramid {
namespace internal {

// Kernel properties

TEST(SeparableFilterTest, KernelSumsToOne) {
    const float sum = std::accumulate(kBurtAdelsonKernel.begin(), kBurtAdelsonKernel.end(), 0.0f);
    EXPECT_NEAR(sum, 1.0f, 1e-6f);
}

TEST(SeparableFilterTest, ReflectIndexBoundaryValuesForSize5) {
    // BORDER_REFLECT_101 mapping for length-5 index range [0,4]; period = 8.
    constexpr int size = 5;
    EXPECT_EQ(ReflectIndex(-2, size), 2);
    EXPECT_EQ(ReflectIndex(-1, size), 1);
    EXPECT_EQ(ReflectIndex(0, size), 0);
    EXPECT_EQ(ReflectIndex(4, size), 4);
    EXPECT_EQ(ReflectIndex(5, size), 3);
    EXPECT_EQ(ReflectIndex(6, size), 2);
    EXPECT_EQ(ReflectIndex(7, size), 1);
}

TEST(SeparableFilterTest, ExpandKernelIsDoubleBurtAdelson) {
    for (std::size_t i = 0; i < kBurtAdelsonKernel.size(); ++i) {
        EXPECT_NEAR(kExpandKernel[i], kBurtAdelsonKernel[i] * 2.0f, 1e-7f) << "i=" << i;
    }
    const float sum = std::accumulate(kExpandKernel.begin(), kExpandKernel.end(), 0.0f);
    EXPECT_NEAR(sum, 2.0f, 1e-6f);
}

TEST(SeparableFilterTest, KernelIsSymmetricAndEqualContribution) {
    // Symmetry: w(-x) == w(x)
    EXPECT_FLOAT_EQ(kBurtAdelsonKernel[0], kBurtAdelsonKernel[4]);
    EXPECT_FLOAT_EQ(kBurtAdelsonKernel[1], kBurtAdelsonKernel[3]);

    // Equal contribution (Burt-Adelson condition): even-offset weights sum to
    // 0.5 and odd-offset weights sum to 0.5.
    // Even offsets {-2,0,+2}: k[0]+k[2]+k[4] = 2a+c = 0.5
    // Odd offsets  {-1,+1}:   k[1]+k[3]       = 2b   = 0.5
    const float even_sum = kBurtAdelsonKernel[0] + kBurtAdelsonKernel[2] + kBurtAdelsonKernel[4];
    const float odd_sum = kBurtAdelsonKernel[1] + kBurtAdelsonKernel[3];
    EXPECT_NEAR(even_sum, 0.5f, 1e-6f);
    EXPECT_NEAR(odd_sum, 0.5f, 1e-6f);
}

// ConvolveRows

TEST(SeparableFilterTest, UniformImageUnchangedAfterConvolveRows) {
    // Kernel sums to 1 → every output pixel == input constant.
    const cv::Mat img(32, 64, CV_32FC1, cv::Scalar(0.7f));
    const cv::Mat result = ConvolveRows(img, kBurtAdelsonKernel);

    ASSERT_EQ(result.size(), img.size());
    for (int y = 0; y < result.rows; ++y) {
        const float* row = result.ptr<float>(y);
        for (int x = 0; x < result.cols; ++x) {
            EXPECT_NEAR(row[x], 0.7f, 1e-5f) << "at (" << y << "," << x << ")";
        }
    }
}

TEST(SeparableFilterTest, ConvolveRowsSingleRowManualValues) {
    // 1×5 image [1,2,3,4,5], hand-computed expected output.
    // k = {0.05, 0.25, 0.4, 0.25, 0.05}, reflect-101 borders.
    //
    // x=0: src[2,1,0,1,2]={3,2,1,2,3} → 0.05*3+0.25*2+0.4*1+0.25*2+0.05*3 = 1.70
    // x=1: src[1,0,1,2,3]={2,1,2,3,4} → 0.05*2+0.25*1+0.4*2+0.25*3+0.05*4 = 2.10
    // x=2: src[0,1,2,3,4]={1,2,3,4,5} → 0.05*1+0.25*2+0.4*3+0.25*4+0.05*5 = 3.00
    // x=3: src[1,2,3,4,3]={2,3,4,5,4} → 0.05*2+0.25*3+0.4*4+0.25*5+0.05*4 = 3.90
    // x=4: src[2,3,4,3,2]={3,4,5,4,3} → 0.05*3+0.25*4+0.4*5+0.25*4+0.05*3 = 4.30
    const float src_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    const cv::Mat src(1, 5, CV_32FC1, const_cast<float*>(src_data));
    const cv::Mat result = ConvolveRows(src, kBurtAdelsonKernel);

    ASSERT_EQ(result.rows, 1);
    ASSERT_EQ(result.cols, 5);
    const float* row = result.ptr<float>(0);
    EXPECT_NEAR(row[0], 1.70f, 1e-5f) << "x=0";
    EXPECT_NEAR(row[1], 2.10f, 1e-5f) << "x=1";
    EXPECT_NEAR(row[2], 3.00f, 1e-5f) << "x=2";
    EXPECT_NEAR(row[3], 3.90f, 1e-5f) << "x=3";
    EXPECT_NEAR(row[4], 4.30f, 1e-5f) << "x=4";
}

TEST(SeparableFilterTest, ConvolveRowsMirrorInputGivesMirrorOutput) {
    // A horizontally flipped input through a symmetric kernel with symmetric
    // border handling must produce a horizontally flipped output.
    cv::RNG rng(42);
    const int rows = 4;
    const int cols = 32;
    cv::Mat a(rows, cols, CV_32FC1);
    rng.fill(a, cv::RNG::UNIFORM, 0.0f, 1.0f);

    cv::Mat a_flip;
    cv::flip(a, a_flip, 1);  // horizontal flip

    const cv::Mat out_a = ConvolveRows(a, kBurtAdelsonKernel);
    const cv::Mat out_a_flip = ConvolveRows(a_flip, kBurtAdelsonKernel);

    cv::Mat out_a_flipped;
    cv::flip(out_a, out_a_flipped, 1);

    for (int y = 0; y < rows; ++y) {
        const float* row_got = out_a_flip.ptr<float>(y);
        const float* row_expected = out_a_flipped.ptr<float>(y);
        for (int x = 0; x < cols; ++x) {
            EXPECT_NEAR(row_got[x], row_expected[x], 1e-5f) << "at (" << y << "," << x << ")";
        }
    }
}

// GaussianBlur (Faza 3)

TEST(SeparableFilterTest, UniformImageUnchangedAfterGaussianBlur) {
    // Both row and column passes leave a constant image unchanged (kernel sums
    // to 1 in each dimension).
    const cv::Mat img(64, 64, CV_32FC1, cv::Scalar(0.6f));
    const cv::Mat result = GaussianBlur(img);

    ASSERT_EQ(result.size(), img.size());
    for (int y = 0; y < result.rows; ++y) {
        const float* row = result.ptr<float>(y);
        for (int x = 0; x < result.cols; ++x) {
            EXPECT_NEAR(row[x], 0.6f, 1e-5f) << "at (" << y << "," << x << ")";
        }
    }
}

TEST(SeparableFilterTest, ImpulseResponseMatchesKernel) {
    // A single non-zero pixel at the image centre, far from any border, must
    // produce exactly the 2D outer product k⊗k after GaussianBlur.
    const int sz = 16;
    const int cy = 8, cx = 8;  // centre — at least 2 px from every border
    cv::Mat img = cv::Mat::zeros(sz, sz, CV_32FC1);
    img.ptr<float>(cy)[cx] = 1.0f;

    const cv::Mat result = GaussianBlur(img);

    for (int dy = -2; dy <= 2; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
            const float expected = kBurtAdelsonKernel[static_cast<std::size_t>(dy + 2)] *
                                   kBurtAdelsonKernel[static_cast<std::size_t>(dx + 2)];
            const float got = result.ptr<float>(cy + dy)[cx + dx];
            EXPECT_NEAR(got, expected, 1e-6f) << "at dy=" << dy << " dx=" << dx;
        }
    }

    // Every pixel outside the 5×5 support must be zero.
    for (int y = 0; y < sz; ++y) {
        const float* row = result.ptr<float>(y);
        for (int x = 0; x < sz; ++x) {
            if (std::abs(y - cy) > 2 || std::abs(x - cx) > 2) {
                EXPECT_NEAR(row[x], 0.0f, 1e-6f)
                    << "expected zero outside support at (" << y << "," << x << ")";
            }
        }
    }
}

TEST(SeparableFilterTest, GaussianBlurMatchesNaive2DConvolution) {
    // Build the 5×5 2D kernel as the outer product k⊗k, then apply it as a
    // naive direct convolution with reflect-101 border handling.  The result
    // must match GaussianBlur() to floating-point precision.
    cv::RNG rng(42);
    const int rows = 32, cols = 32;
    cv::Mat src(rows, cols, CV_32FC1);
    rng.fill(src, cv::RNG::UNIFORM, 0.0f, 1.0f);

    // --- reference: naive 5×5 direct convolution ---
    cv::Mat ref(rows, cols, CV_32FC1);
    for (int y = 0; y < rows; ++y) {
        float* ref_row = ref.ptr<float>(y);
        for (int x = 0; x < cols; ++x) {
            float val = 0.0f;
            for (int dy = -2; dy <= 2; ++dy) {
                const float* src_r = src.ptr<float>(ReflectIndex(y + dy, rows));
                const float ky = kBurtAdelsonKernel[static_cast<std::size_t>(dy + 2)];
                for (int dx = -2; dx <= 2; ++dx) {
                    val += ky * kBurtAdelsonKernel[static_cast<std::size_t>(dx + 2)] *
                           src_r[ReflectIndex(x + dx, cols)];
                }
            }
            ref_row[x] = val;
        }
    }

    // --- separable ---
    const cv::Mat result = GaussianBlur(src);

    for (int y = 0; y < rows; ++y) {
        const float* row_res = result.ptr<float>(y);
        const float* row_ref = ref.ptr<float>(y);
        for (int x = 0; x < cols; ++x) {
            EXPECT_NEAR(row_res[x], row_ref[x], 1e-5f) << "at (" << y << "," << x << ")";
        }
    }
}

}  // namespace internal
}  // namespace pyramid
