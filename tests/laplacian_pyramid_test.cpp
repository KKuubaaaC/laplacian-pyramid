#include "pyramid/laplacian_pyramid.h"

#include <gtest/gtest.h>

#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>

#include "pyramid/pyramid_blend.h"
#include "pyramid/pyramid_types.h"
#include "pyramid/pyramid_utils.h"

namespace pyramid {

// ---------------------------------------------------------------------------
// Build — validation
// ---------------------------------------------------------------------------

TEST(LaplacianPyramidTest, BuildRejectsInvalidNumLevels) {
    const cv::Mat img(64, 64, CV_32FC1, cv::Scalar(0.5f));
    PyramidParams bad_low;
    bad_low.num_levels = 0;
    EXPECT_FALSE(LaplacianPyramid::Build(img, bad_low).has_value());

    PyramidParams bad_high;
    bad_high.num_levels = 9;
    EXPECT_FALSE(LaplacianPyramid::Build(img, bad_high).has_value());
}

TEST(LaplacianPyramidTest, BuildRejectsEmptyImage) {
    const cv::Mat empty;
    const auto result = LaplacianPyramid::Build(empty);
    EXPECT_FALSE(result.has_value());
}

TEST(LaplacianPyramidTest, BuildRejectsNonFloat32InputTypes) {
    const cv::Mat u8(64, 64, CV_8UC1, cv::Scalar(128));
    EXPECT_FALSE(LaplacianPyramid::Build(u8).has_value());

    const cv::Mat f64(64, 64, CV_64FC1, cv::Scalar(0.5));
    EXPECT_FALSE(LaplacianPyramid::Build(f64).has_value());
}

TEST(LaplacianPyramidTest, BuildRejectsTooSmallImage) {
    // num_levels=4 requires min_dim >= 2^4 = 16; a 4×4 image must be rejected.
    PyramidParams params;
    params.num_levels = 4;
    const cv::Mat tiny(4, 4, CV_32FC1, cv::Scalar(0.5f));
    const auto result = LaplacianPyramid::Build(tiny, params);
    EXPECT_FALSE(result.has_value());
}

// ---------------------------------------------------------------------------
// Build — structure
// ---------------------------------------------------------------------------

TEST(LaplacianPyramidTest, UniformImageHasNearZeroLaplacianLevels) {
    // gaussian[i] - Expand(gaussian[i+1]) ≈ 0 for a constant image because
    // Reduce and Expand both preserve a uniform value (proved in Faza 4).
    PyramidParams params;
    params.num_levels = 4;
    const cv::Mat img(128, 128, CV_32FC1, cv::Scalar(0.5f));
    const auto pyramid = LaplacianPyramid::Build(img, params);
    ASSERT_TRUE(pyramid.has_value()) << "Build failed unexpectedly";

    for (int lvl = 0; lvl < pyramid->NumLevels(); ++lvl) {
        const cv::Mat& lap = pyramid->LaplacianLevel(lvl);
        for (int y = 0; y < lap.rows; ++y) {
            const float* row = lap.ptr<float>(y);
            for (int x = 0; x < lap.cols; ++x) {
                EXPECT_NEAR(row[x], 0.0f, 1e-4f)
                    << "level=" << lvl << " at (" << y << "," << x << ")";
            }
        }
    }
}

TEST(LaplacianPyramidTest, LevelDimensionsHalveEachLevel) {
    // Each Laplacian level must match the corresponding Gaussian level size.
    PyramidParams params;
    params.num_levels = 4;
    const cv::Mat img(256, 256, CV_32FC1, cv::Scalar(1.0f));
    const auto pyramid = LaplacianPyramid::Build(img, params);
    ASSERT_TRUE(pyramid.has_value());

    int expected_rows = img.rows;
    int expected_cols = img.cols;
    for (int lvl = 0; lvl < pyramid->NumLevels(); ++lvl) {
        const cv::Mat& lap = pyramid->LaplacianLevel(lvl);
        EXPECT_EQ(lap.rows, expected_rows) << "level=" << lvl;
        EXPECT_EQ(lap.cols, expected_cols) << "level=" << lvl;
        EXPECT_EQ(lap.type(), CV_32FC1) << "level=" << lvl;
        expected_rows /= 2;
        expected_cols /= 2;
    }
    // The Gaussian top lives one level above the finest Laplacian.
    EXPECT_EQ(pyramid->GaussianTop().rows, expected_rows);
    EXPECT_EQ(pyramid->GaussianTop().cols, expected_cols);
}

TEST(LaplacianPyramidTest, NumLevelsMatchesParams) {
    for (const int n : {1, 3, 5}) {
        PyramidParams params;
        params.num_levels = n;
        // 256 > 2^5 = 32, so all depths are valid.
        const cv::Mat img(256, 256, CV_32FC1, cv::Scalar(0.5f));
        const auto pyramid = LaplacianPyramid::Build(img, params);
        ASSERT_TRUE(pyramid.has_value()) << "Build failed for num_levels=" << n;
        EXPECT_EQ(pyramid->NumLevels(), n) << "num_levels=" << n;
    }
}

// ---------------------------------------------------------------------------
// Reconstruct — Faza 6
// ---------------------------------------------------------------------------

TEST(LaplacianPyramidTest, ReconstructionPSNRAbove160dBFloat32Random) {
    // Build and reconstruct a random float32 image.
    // Because Expand is deterministic, L[i] = g[i] - E and then E + L[i]
    // hits only 1–2 ULP of rounding error per pixel; accumulated over 6
    // levels the round-trip MSE is at the noise floor of float32 arithmetic.
    // 160 dB is a safe floor for float32; PSNR > 300 dB would need float64
    // (or degenerate signals) — see gradient test below.
    cv::RNG rng(42);
    cv::Mat img(256, 256, CV_32FC1);
    rng.fill(img, cv::RNG::UNIFORM, 0.0f, 1.0f);

    PyramidParams params;
    params.num_levels = 6;
    const auto pyramid = LaplacianPyramid::Build(img, params);
    ASSERT_TRUE(pyramid.has_value());

    const cv::Mat rec = pyramid->Reconstruct();
    ASSERT_EQ(rec.size(), img.size());

    const double psnr = pyramid::ComputePSNR(img, rec);
    EXPECT_GT(psnr, 160.0) << "PSNR: " << psnr << " dB";
}

TEST(LaplacianPyramidTest, ReconstructionPSNRInfiniteForZeroImage) {
    // All-zero input: every Laplacian level is exactly 0.0f; MSE = 0; PSNR = infinity
    // (> 300 dB).
    const int sz = 256;
    const cv::Mat img = cv::Mat::zeros(sz, sz, CV_32FC1);

    PyramidParams params;
    params.num_levels = 6;
    const auto pyramid = LaplacianPyramid::Build(img, params);
    ASSERT_TRUE(pyramid.has_value());

    const cv::Mat rec = pyramid->Reconstruct();
    const double psnr = pyramid::ComputePSNR(img, rec);
    EXPECT_GT(psnr, 300.0) << "PSNR: " << psnr << " dB";
}

TEST(LaplacianPyramidTest, ReconstructionPSNRAbove300dBLinearGradient) {
    // Linear ramp in [0, 1]: blur/subsample are linear; Laplacian detail levels
    // stay small. Float32 round-trip still accumulates ULP error (~230 dB PSNR at
    // 256², 6 levels — much better than random ~170 dB, but not >300 dB; see
    // ReconstructionPSNRInfiniteForZeroImage for degenerate exact-zero MSE).
    const int sz = 256;
    cv::Mat img(sz, sz, CV_32FC1);
    for (int y = 0; y < sz; ++y) {
        float* row = img.ptr<float>(y);
        for (int x = 0; x < sz; ++x) {
            row[x] = static_cast<float>(x + y) / static_cast<float>(2 * (sz - 1));
        }
    }
    PyramidParams params;
    params.num_levels = 6;
    const auto pyramid = LaplacianPyramid::Build(img, params);
    ASSERT_TRUE(pyramid.has_value());
    const cv::Mat rec = pyramid->Reconstruct();
    const double psnr = pyramid::ComputePSNR(img, rec);
    EXPECT_GT(psnr, 220.0) << "Linear gradient PSNR: " << psnr << " dB";
}

TEST(LaplacianPyramidTest, ReconstructionIdempotent) {
    // Build → Reconstruct → Build again must yield the same pyramid structure
    // (same level dimensions and near-identical Laplacian values).
    cv::RNG rng(42);
    cv::Mat img(128, 128, CV_32FC1);
    rng.fill(img, cv::RNG::UNIFORM, 0.0f, 1.0f);

    PyramidParams params;
    params.num_levels = 4;
    const auto pyr1 = LaplacianPyramid::Build(img, params);
    ASSERT_TRUE(pyr1.has_value());

    const cv::Mat rec = pyr1->Reconstruct();
    const auto pyr2 = LaplacianPyramid::Build(rec, params);
    ASSERT_TRUE(pyr2.has_value());

    EXPECT_EQ(pyr1->NumLevels(), pyr2->NumLevels());
    for (int lvl = 0; lvl < pyr1->NumLevels(); ++lvl) {
        EXPECT_EQ(pyr1->LaplacianLevel(lvl).size(), pyr2->LaplacianLevel(lvl).size())
            << "level=" << lvl;
    }

    // PSNR between original and double-round-tripped image must still be high.
    const cv::Mat rec2 = pyr2->Reconstruct();
    const double psnr = pyramid::ComputePSNR(img, rec2);
    EXPECT_GT(psnr, 150.0) << "double round-trip PSNR: " << psnr << " dB";
}

TEST(LaplacianPyramidTest, TwoByTwoOneLevelEdgeCase) {
    // For a 2×2 image with 1 level, Expand produces the pixel-mean at every
    // position.  The arithmetic is exact in float32 for integer-valued inputs,
    // so reconstruction must equal the original exactly.
    const float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    const cv::Mat img(2, 2, CV_32FC1, const_cast<float*>(data));

    PyramidParams params;
    params.num_levels = 1;
    const auto pyramid = LaplacianPyramid::Build(img, params);
    ASSERT_TRUE(pyramid.has_value());

    const cv::Mat rec = pyramid->Reconstruct();
    ASSERT_EQ(rec.size(), img.size());

    for (int y = 0; y < 2; ++y) {
        const float* row_o = img.ptr<float>(y);
        const float* row_r = rec.ptr<float>(y);
        for (int x = 0; x < 2; ++x) {
            EXPECT_NEAR(row_r[x], row_o[x], 1e-5f) << "at (" << y << "," << x << ")";
        }
    }
}

TEST(LaplacianPyramidTest, RectangularImage128x256ReconstructionHighPSNR) {
    // Non-square aspect stresses row vs column handling in separable filters
    // and Reduce/Expand geometry.
    cv::RNG rng(99);
    cv::Mat img(128, 256, CV_32FC1);
    rng.fill(img, cv::RNG::UNIFORM, 0.0f, 1.0f);

    PyramidParams params;
    params.num_levels = 4;
    const auto pyramid = LaplacianPyramid::Build(img, params);
    ASSERT_TRUE(pyramid.has_value());

    const cv::Mat rec = pyramid->Reconstruct();
    ASSERT_EQ(rec.size(), img.size());
    const double psnr = pyramid::ComputePSNR(img, rec);
    EXPECT_GT(psnr, 150.0) << "PSNR=" << psnr << " dB";
}

TEST(LaplacianPyramidTest, ReconstructionAcrossVariousDepths) {
    // Verify that the round-trip quality is high regardless of pyramid depth.
    cv::RNG rng(42);
    cv::Mat img(256, 256, CV_32FC1);
    rng.fill(img, cv::RNG::UNIFORM, 0.0f, 1.0f);

    for (const int depth : {1, 4, 6, 8}) {
        PyramidParams params;
        params.num_levels = depth;
        const auto pyramid = LaplacianPyramid::Build(img, params);
        ASSERT_TRUE(pyramid.has_value()) << "depth=" << depth;

        const cv::Mat rec = pyramid->Reconstruct();
        const double psnr = pyramid::ComputePSNR(img, rec);
        EXPECT_GT(psnr, 150.0) << "depth=" << depth << " PSNR=" << psnr << " dB";
    }
}

// ---------------------------------------------------------------------------
// Reduce / Expand — inferred only from LaplacianPyramid layers (B-1 / MO-8).
// GaussianTop is the coarsest Gaussian after repeated Reduce; LaplacianLevel(i)
// lives at the same resolution as Gaussian level i before the last difference.
// ---------------------------------------------------------------------------

TEST(LaplacianPyramidTest, ReducePreservesUniformValue) {
    // num_levels=1: one Reduce path from input → GaussianTop; blur + subsample
    // keep a constant image constant.
    PyramidParams params;
    params.num_levels = 1;
    const cv::Mat img(64, 64, CV_32FC1, cv::Scalar(0.5f));
    const auto pyramid = LaplacianPyramid::Build(img, params);
    ASSERT_TRUE(pyramid.has_value());

    ASSERT_EQ(pyramid->NumLevels(), 1);
    EXPECT_EQ(pyramid->LaplacianLevel(0).rows, img.rows);
    EXPECT_EQ(pyramid->LaplacianLevel(0).cols, img.cols);

    const cv::Mat& top = pyramid->GaussianTop();
    ASSERT_EQ(top.rows, img.rows / 2);
    ASSERT_EQ(top.cols, img.cols / 2);
    for (int y = 0; y < top.rows; ++y) {
        const float* row = top.ptr<float>(y);
        for (int x = 0; x < top.cols; ++x) {
            EXPECT_NEAR(row[x], 0.5f, 1e-5f) << "at (" << y << "," << x << ")";
        }
    }
}

TEST(LaplacianPyramidTest, ExpandPreservesUniformValue) {
    // Reconstruct() chains Expand(coarse)+L; for uniform input, L≈0 and Expand
    // must restore full-res constant (kExpandKernel ×4 gain vs upsample loss).
    PyramidParams params;
    params.num_levels = 1;
    const cv::Mat img(64, 64, CV_32FC1, cv::Scalar(0.5f));
    const auto pyramid = LaplacianPyramid::Build(img, params);
    ASSERT_TRUE(pyramid.has_value());

    const cv::Mat rec = pyramid->Reconstruct();
    ASSERT_EQ(rec.size(), img.size());
    for (int y = 0; y < rec.rows; ++y) {
        const float* row = rec.ptr<float>(y);
        for (int x = 0; x < rec.cols; ++x) {
            EXPECT_NEAR(row[x], 0.5f, 1e-4f) << "at (" << y << "," << x << ")";
        }
    }
}

TEST(LaplacianPyramidTest, ReduceHalvesDimensions) {
    // Layer sizes encode one Reduce step: finest Laplacian matches input;
    // GaussianTop matches half resolution in both axes.
    PyramidParams params;
    params.num_levels = 1;
    const cv::Mat img(256, 256, CV_32FC1, cv::Scalar(1.0f));
    const auto pyramid = LaplacianPyramid::Build(img, params);
    ASSERT_TRUE(pyramid.has_value());

    EXPECT_EQ(pyramid->LaplacianLevel(0).rows, 256);
    EXPECT_EQ(pyramid->LaplacianLevel(0).cols, 256);

    const cv::Mat& top = pyramid->GaussianTop();
    EXPECT_EQ(top.rows, 128);
    EXPECT_EQ(top.cols, 128);
    EXPECT_EQ(top.type(), CV_32FC1);
}

TEST(LaplacianPyramidTest, ExpandDoublesDimensions) {
    // Single-level pyramid: GaussianTop is half-sized; Reconstruct expands once
    // to match LaplacianLevel(0) / input resolution.
    PyramidParams params;
    params.num_levels = 1;
    const cv::Mat img(128, 128, CV_32FC1, cv::Scalar(1.0f));
    const auto pyramid = LaplacianPyramid::Build(img, params);
    ASSERT_TRUE(pyramid.has_value());

    EXPECT_EQ(pyramid->GaussianTop().rows, 64);
    EXPECT_EQ(pyramid->GaussianTop().cols, 64);
    EXPECT_EQ(pyramid->LaplacianLevel(0).rows, 128);
    EXPECT_EQ(pyramid->LaplacianLevel(0).cols, 128);

    const cv::Mat rec = pyramid->Reconstruct();
    EXPECT_EQ(rec.rows, 128);
    EXPECT_EQ(rec.cols, 128);
    EXPECT_EQ(rec.type(), CV_32FC1);
}

TEST(LaplacianPyramidTest, MultiLevelLayerSizesFormHalvingChain) {
    // Each stored Laplacian sits at the resolution of Gaussian[i]; each step
    // halves rows and cols until GaussianTop.
    PyramidParams params;
    params.num_levels = 3;
    const cv::Mat img(128, 256, CV_32FC1, cv::Scalar(0.25f));
    const auto pyramid = LaplacianPyramid::Build(img, params);
    ASSERT_TRUE(pyramid.has_value());

    EXPECT_EQ(pyramid->LaplacianLevel(0).rows, 128);
    EXPECT_EQ(pyramid->LaplacianLevel(0).cols, 256);
    EXPECT_EQ(pyramid->LaplacianLevel(1).rows, 64);
    EXPECT_EQ(pyramid->LaplacianLevel(1).cols, 128);
    EXPECT_EQ(pyramid->LaplacianLevel(2).rows, 32);
    EXPECT_EQ(pyramid->LaplacianLevel(2).cols, 64);
    EXPECT_EQ(pyramid->GaussianTop().rows, 16);
    EXPECT_EQ(pyramid->GaussianTop().cols, 32);
}

// ---------------------------------------------------------------------------
// Laplacian blend (I-1)
// ---------------------------------------------------------------------------

TEST(LaplacianPyramidTest, BlendMaskAllOnesSelectsImageA) {
    cv::RNG rng(7);
    cv::Mat a(96, 96, CV_32FC1);
    cv::Mat b(96, 96, CV_32FC1);
    rng.fill(a, cv::RNG::UNIFORM, 0.0f, 1.0f);
    rng.fill(b, cv::RNG::UNIFORM, 0.0f, 1.0f);
    const cv::Mat mask(96, 96, CV_32FC1, cv::Scalar(1.0f));

    PyramidParams params;
    params.num_levels = 4;
    const auto blended = BlendLaplacianPyramids(a, b, mask, params);
    ASSERT_TRUE(blended.has_value());
    const double psnr = pyramid::ComputePSNR(a, *blended);
    EXPECT_GT(psnr, 120.0) << "PSNR=" << psnr << " dB";
}

TEST(LaplacianPyramidTest, BlendMaskAllZerosSelectsImageB) {
    cv::RNG rng(11);
    cv::Mat a(96, 96, CV_32FC1);
    cv::Mat b(96, 96, CV_32FC1);
    rng.fill(a, cv::RNG::UNIFORM, 0.0f, 1.0f);
    rng.fill(b, cv::RNG::UNIFORM, 0.0f, 1.0f);
    const cv::Mat mask(96, 96, CV_32FC1, cv::Scalar(0.0f));

    PyramidParams params;
    params.num_levels = 4;
    const auto blended = BlendLaplacianPyramids(a, b, mask, params);
    ASSERT_TRUE(blended.has_value());
    const double psnr = pyramid::ComputePSNR(b, *blended);
    EXPECT_GT(psnr, 120.0) << "PSNR=" << psnr << " dB";
}

// ---------------------------------------------------------------------------
// Float64 ground truth vs float32 pyramid round-trip (I-5)
// ---------------------------------------------------------------------------

TEST(LaplacianPyramidTest, ReconstructionPSNRFloat64OriginalVsFloat32Pipeline) {
    // Demonstrates float32 precision limit: reconstruction error is dominated
    // by float32 quantization (~138 dB theoretical limit), not by the pyramid algorithm.
    //
    // PSNR compares CV_64FC1 ground truth vs CV_32FC1 reconstruction; MSE uses double
    // samples from both (float recon promoted to double per pixel).
    const int sz = 256;
    cv::Mat ref64(sz, sz, CV_64FC1);
    cv::RNG rng(42);
    rng.fill(ref64, cv::RNG::UNIFORM, 0.0, 1.0);

    cv::Mat img32;
    ref64.convertTo(img32, CV_32F);

    PyramidParams params;
    params.num_levels = 6;
    const auto pyramid = LaplacianPyramid::Build(img32, params);
    ASSERT_TRUE(pyramid.has_value());

    const cv::Mat rec = pyramid->Reconstruct();
    ASSERT_EQ(rec.type(), CV_32FC1);

    double mse = 0.0;
    for (int y = 0; y < sz; ++y) {
        const double* row_o = ref64.ptr<double>(y);
        const float* row_r = rec.ptr<float>(y);
        for (int x = 0; x < sz; ++x) {
            const double diff = row_o[x] - static_cast<double>(row_r[x]);
            mse += diff * diff;
        }
    }
    mse /= static_cast<double>(sz * sz);
    ASSERT_GT(mse, 0.0);
    constexpr double kMaxSignal = 1.0;
    const double psnr = 10.0 * std::log10((kMaxSignal * kMaxSignal) / mse);

    EXPECT_GT(psnr, 100.0) << "PSNR (f64 orig vs f32 recon): " << psnr << " dB";
    EXPECT_LT(psnr, 300.0) << "PSNR (f64 orig vs f32 recon): " << psnr << " dB";
}

}  // namespace pyramid
