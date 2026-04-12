#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "pyramid/laplacian_pyramid.h"
#include "pyramid/pyramid_blend.h"
#include "pyramid/pyramid_types.h"
#include "pyramid/pyramid_utils.h"

namespace {

constexpr int kThumbMaxSide = 256;

// Laplacian visualization: zero → 128, scale by per-layer max |value|, clamp [0,255]
[[nodiscard]] cv::Mat LaplacianToU8(const cv::Mat& lap) {
    CV_Assert(lap.type() == CV_32FC1);
    double min_val = 0.0;
    double max_val = 0.0;
    cv::minMaxLoc(lap, &min_val, &max_val);
    const double abs_max = std::max(std::fabs(min_val), std::fabs(max_val));

    cv::Mat u8(lap.rows, lap.cols, CV_8U);
    if (abs_max < 1e-9) {
        u8.setTo(128);
        return u8;
    }

    for (int y = 0; y < lap.rows; ++y) {
        const float* src_row = lap.ptr<float>(y);
        uchar* dst_row = u8.ptr<uchar>(y);
        for (int x = 0; x < lap.cols; ++x) {
            const float t = static_cast<float>(src_row[x] / abs_max);
            int v = static_cast<int>(std::lround(t * 127.0f + 128.0f));
            v = std::clamp(v, 0, 255);
            dst_row[x] = static_cast<uchar>(v);
        }
    }
    return u8;
}

// Loads image as CV_32F in [0,1]: CV_32FC1 (gray) or CV_32FC3 (BGR). Returns false if read fails
[[nodiscard]] bool LoadFloat01(const std::string& path, cv::Mat& out, bool* is_color) {
    cv::Mat raw = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (raw.empty()) {
        return false;
    }
    if (raw.channels() == 4) {
        cv::cvtColor(raw, raw, cv::COLOR_BGRA2BGR);
    }
    if (raw.channels() == 1) {
        raw.convertTo(out, CV_32F, 1.0 / 255.0);
        *is_color = false;
        return true;
    }
    if (raw.channels() == 3) {
        raw.convertTo(out, CV_32FC3, 1.0 / 255.0);
        *is_color = true;
        return true;
    }
    return false;
}

[[nodiscard]] bool LoadMaskFloat01(const std::string& path, cv::Mat& out) {
    cv::Mat raw = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (raw.empty()) {
        return false;
    }
    raw.convertTo(out, CV_32F, 1.0 / 255.0);
    return true;
}

[[nodiscard]] bool SaveReconstructionPng(const cv::Mat& recon_f32_01,
                                         const std::filesystem::path& path) {
    cv::Mat u8;
    if (recon_f32_01.channels() == 1) {
        recon_f32_01.convertTo(u8, CV_8U, 255.0);
    } else {
        recon_f32_01.convertTo(u8, CV_8UC3, 255.0);
    }
    if (!cv::imwrite(path.string(), u8)) {
        std::cerr << "Error: failed to write " << path << "\n";
        return false;
    }
    return true;
}

// Resize thumbnail preserving aspect; max side = thumb_max_side
[[nodiscard]] cv::Mat ResizeThumb(const cv::Mat& u8, int thumb_max_side) {
    const int h = u8.rows;
    const int w = u8.cols;
    const int big = std::max(h, w);
    if (big <= thumb_max_side) {
        return u8.clone();
    }
    const double scale = static_cast<double>(thumb_max_side) / static_cast<double>(big);
    cv::Mat resized;
    cv::resize(u8, resized, cv::Size(), scale, scale, cv::INTER_AREA);
    return resized;
}

// Grid mosaic: each cell padded to (cell_h × cell_w), centered
[[nodiscard]] cv::Mat MakeMosaic(const std::vector<cv::Mat>& thumbs, int cell_w, int cell_h) {
    const int n = static_cast<int>(thumbs.size());
    if (n == 0) {
        return {};
    }
    const int cols = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(n))));
    const int rows = (n + cols - 1) / cols;

    const int type = thumbs[0].channels() == 3 ? CV_8UC3 : CV_8U;
    cv::Mat mosaic(rows * cell_h, cols * cell_w, type, cv::Scalar(64));
    for (int i = 0; i < n; ++i) {
        const int gr = i / cols;
        const int gc = i % cols;
        const cv::Mat& t = thumbs[static_cast<std::size_t>(i)];
        const int off_x = gc * cell_w + (cell_w - t.cols) / 2;
        const int off_y = gr * cell_h + (cell_h - t.rows) / 2;
        t.copyTo(mosaic(cv::Rect(off_x, off_y, t.cols, t.rows)));
    }
    return mosaic;
}

[[nodiscard]] std::optional<int> ParseNumLevels(int argc, char** argv, int default_levels,
                                                int arg_index) {
    if (argc <= arg_index) {
        return default_levels;
    }
    try {
        const std::string arg(argv[arg_index]);
        std::size_t idx = 0;
        const int parsed = std::stoi(arg, &idx);
        while (idx < arg.size() && std::isspace(static_cast<unsigned char>(arg[idx])) != 0) {
            ++idx;
        }
        if (idx != arg.size() || parsed < 1 || parsed > 8) {
            return std::nullopt;
        }
        return parsed;
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

void PrintNumLevelsParseError(int argc, char** argv, int arg_index) {
    std::cerr << "Error: num_levels must be an integer from 1 to 8";
    if (argc > arg_index) {
        std::cerr << ", got: " << argv[arg_index];
    }
    std::cerr << "\n";
}

void PrintPyramidBenchmarkLine(const cv::Mat& gray01, int num_levels) {
    using clock = std::chrono::steady_clock;
    pyramid::PyramidParams params;
    params.num_levels = num_levels;

    const auto t0 = clock::now();
    const auto lap_pyr = pyramid::LaplacianPyramid::Build(gray01, params);
    const auto t1 = clock::now();
    if (!lap_pyr.has_value()) {
        return;
    }
    const cv::Mat reconstructed = lap_pyr->Reconstruct();
    (void)reconstructed;
    const auto t2 = clock::now();

    std::vector<cv::Mat> opencv_gaussian;
    const auto t3 = clock::now();
    cv::buildPyramid(gray01, opencv_gaussian, num_levels);
    const auto t4 = clock::now();

    const double ms_build = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double ms_recon = std::chrono::duration<double, std::milli>(t2 - t1).count();
    const double ms_opencv = std::chrono::duration<double, std::milli>(t4 - t3).count();

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Timing (ms): Laplacian Build=" << ms_build << " Reconstruct=" << ms_recon
              << " (sum=" << (ms_build + ms_recon) << "); OpenCV buildPyramid (levels 0.."
              << num_levels << ")=" << ms_opencv << "\n";
}

[[nodiscard]] bool RunGrayscaleDemo(const cv::Mat& input, int num_levels,
                                    const std::filesystem::path& out_dir) {
    pyramid::PyramidParams params;
    params.num_levels = num_levels;
    auto pyramid_opt = pyramid::LaplacianPyramid::Build(input, params);
    if (!pyramid_opt.has_value()) {
        std::cerr << "Error: LaplacianPyramid::Build failed (empty, wrong type, or image "
                     "too small for num_levels)\n";
        return false;
    }
    pyramid::LaplacianPyramid pyramid = std::move(*pyramid_opt);

    const cv::Mat reconstructed = pyramid.Reconstruct();
    if (reconstructed.size() != input.size() || reconstructed.type() != CV_32FC1) {
        std::cerr << "Error: Reconstruct size/type mismatch\n";
        return false;
    }

    const double psnr = pyramid::ComputePSNR(input, reconstructed);
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "PSNR (reconstruction vs original, MAX=1.0): " << psnr << " dB\n";

    PrintPyramidBenchmarkLine(input, num_levels);

    for (int i = 0; i < pyramid.NumLevels(); ++i) {
        const cv::Mat u8 = LaplacianToU8(pyramid.LaplacianLevel(i));
        std::ostringstream name;
        name << "laplacian_" << std::setw(2) << std::setfill('0') << i << ".png";
        const std::filesystem::path out_path = out_dir / name.str();
        if (!cv::imwrite(out_path.string(), u8)) {
            std::cerr << "Error: failed to write " << out_path << "\n";
            return false;
        }
        std::cout << "Wrote " << out_path << "\n";
    }

    const std::filesystem::path recon_path = out_dir / "reconstruction.png";
    if (!SaveReconstructionPng(reconstructed, recon_path)) {
        return false;
    }
    std::cout << "Wrote " << recon_path << "\n";

    std::vector<cv::Mat> thumbs;
    const int num_lap_levels = pyramid.NumLevels();
    thumbs.reserve(static_cast<std::size_t>(num_lap_levels) + 1u);
    for (int i = 0; i < num_lap_levels; ++i) {
        thumbs.push_back(ResizeThumb(LaplacianToU8(pyramid.LaplacianLevel(i)), kThumbMaxSide));
    }
    {
        const cv::Mat& gtop = pyramid.GaussianTop();
        cv::Mat g8;
        double gmax = 0.0;
        cv::minMaxLoc(gtop, nullptr, &gmax);
        if (gmax <= 1.01) {
            gtop.convertTo(g8, CV_8U, 255.0);
        } else {
            gtop.convertTo(g8, CV_8U);
        }
        thumbs.push_back(ResizeThumb(g8, kThumbMaxSide));
    }

    int cell_w = 0;
    int cell_h = 0;
    for (const cv::Mat& t : thumbs) {
        cell_w = std::max(cell_w, t.cols);
        cell_h = std::max(cell_h, t.rows);
    }
    cell_w += 8;
    cell_h += 8;

    const cv::Mat mosaic = MakeMosaic(thumbs, cell_w, cell_h);
    if (!mosaic.empty()) {
        const std::filesystem::path mosaic_path = out_dir / "mosaic.png";
        if (!cv::imwrite(mosaic_path.string(), mosaic)) {
            std::cerr << "Error: failed to write " << mosaic_path << "\n";
            return false;
        }
        std::cout << "Wrote " << mosaic_path << "\n";
    }
    return true;
}

[[nodiscard]] bool RunColorDemo(const cv::Mat& input_bgr01, int num_levels,
                                const std::filesystem::path& out_dir) {
    std::vector<cv::Mat> ch(3);
    cv::split(input_bgr01, ch);

    pyramid::PyramidParams params;
    params.num_levels = num_levels;

    double psnr_sum = 0.0;
    std::vector<cv::Mat> recon_ch(3);

    for (int c = 0; c < 3; ++c) {
        auto pyramid_opt =
            pyramid::LaplacianPyramid::Build(ch[static_cast<std::size_t>(c)], params);
        if (!pyramid_opt.has_value()) {
            std::cerr << "Error: LaplacianPyramid::Build failed on channel " << c << "\n";
            return false;
        }
        pyramid::LaplacianPyramid pyramid = std::move(*pyramid_opt);
        recon_ch[static_cast<std::size_t>(c)] = pyramid.Reconstruct();

        for (int i = 0; i < pyramid.NumLevels(); ++i) {
            const cv::Mat u8 = LaplacianToU8(pyramid.LaplacianLevel(i));
            std::ostringstream name;
            name << "laplacian_" << std::setw(2) << std::setfill('0') << i << "_ch" << c << ".png";
            const std::filesystem::path out_path = out_dir / name.str();
            if (!cv::imwrite(out_path.string(), u8)) {
                std::cerr << "Error: failed to write " << out_path << "\n";
                return false;
            }
        }
        psnr_sum += pyramid::ComputePSNR(ch[static_cast<std::size_t>(c)],
                                         recon_ch[static_cast<std::size_t>(c)]);
    }

    cv::Mat reconstructed;
    cv::merge(recon_ch, reconstructed);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "PSNR mean over B,G,R channels (MAX=1.0): " << (psnr_sum / 3.0) << " dB\n";

    // Benchmark on luminance-like average (single 32F plane).
    cv::Mat gray01;
    cv::cvtColor(input_bgr01, gray01, cv::COLOR_BGR2GRAY);
    PrintPyramidBenchmarkLine(gray01, num_levels);

    const std::filesystem::path recon_path = out_dir / "reconstruction.png";
    if (!SaveReconstructionPng(reconstructed, recon_path)) {
        return false;
    }
    std::cout << "Wrote " << recon_path << "\n";
    return true;
}

[[nodiscard]] bool RunBlendDemo(const std::string& path_a, const std::string& path_b,
                                const std::string& path_mask, int num_levels,
                                const std::filesystem::path& out_dir) {
    cv::Mat a;
    cv::Mat b;
    cv::Mat mask;
    bool color_a = false;
    bool color_b = false;
    if (!LoadFloat01(path_a, a, &color_a) || !LoadFloat01(path_b, b, &color_b)) {
        std::cerr << "Error: could not read blend inputs\n";
        return false;
    }
    if (color_a != color_b) {
        std::cerr << "Error: blend inputs must be both grayscale or both color\n";
        return false;
    }
    if (!LoadMaskFloat01(path_mask, mask)) {
        std::cerr << "Error: could not read mask\n";
        return false;
    }
    if (a.size() != b.size() || a.size() != mask.size()) {
        std::cerr << "Error: A, B, and mask must have the same dimensions\n";
        return false;
    }

    pyramid::PyramidParams params;
    params.num_levels = num_levels;

    if (!color_a) {
        const auto blended = pyramid::BlendLaplacianPyramids(a, b, mask, params);
        if (!blended.has_value()) {
            std::cerr << "Error: BlendLaplacianPyramids failed\n";
            return false;
        }
        const std::filesystem::path out_path = out_dir / "blended.png";
        if (!SaveReconstructionPng(*blended, out_path)) {
            return false;
        }
        std::cout << "Wrote " << out_path << "\n";
        return true;
    }

    std::vector<cv::Mat> ca(3);
    std::vector<cv::Mat> cb(3);
    cv::split(a, ca);
    cv::split(b, cb);
    std::vector<cv::Mat> out_ch(3);
    for (int c = 0; c < 3; ++c) {
        const auto blended = pyramid::BlendLaplacianPyramids(
            ca[static_cast<std::size_t>(c)], cb[static_cast<std::size_t>(c)], mask, params);
        if (!blended.has_value()) {
            std::cerr << "Error: BlendLaplacianPyramids failed on channel " << c << "\n";
            return false;
        }
        out_ch[static_cast<std::size_t>(c)] = *blended;
    }
    cv::Mat merged;
    cv::merge(out_ch, merged);
    const std::filesystem::path out_path = out_dir / "blended.png";
    if (!SaveReconstructionPng(merged, out_path)) {
        return false;
    }
    std::cout << "Wrote " << out_path << "\n";
    return true;
}

}  // namespace

int main(int argc, char* argv[]) {
    const std::filesystem::path out_dir = "output";
    std::error_code ec;
    std::filesystem::create_directories(out_dir, ec);
    if (ec) {
        std::cerr << "Error: cannot create output directory: " << ec.message() << "\n";
        return EXIT_FAILURE;
    }

    if (argc >= 2 && std::string(argv[1]) == "--blend") {
        if (argc < 5) {
            std::cerr << "Usage: " << argv[0]
                      << " --blend <image_a> <image_b> <mask_grayscale> [num_levels]\n";
            return EXIT_FAILURE;
        }
        const std::string& path_a = argv[2];
        const std::string& path_b = argv[3];
        const std::string& path_mask = argv[4];
        const auto num_levels_opt = ParseNumLevels(argc, argv, 6, 5);
        if (!num_levels_opt.has_value()) {
            PrintNumLevelsParseError(argc, argv, 5);
            return EXIT_FAILURE;
        }
        const int num_levels = *num_levels_opt;
        if (!RunBlendDemo(path_a, path_b, path_mask, num_levels, out_dir)) {
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path> [num_levels]\n";
        std::cerr << "       " << argv[0]
                  << " --blend <image_a> <image_b> <mask_grayscale> [num_levels]\n";
        return EXIT_FAILURE;
    }

    const std::string image_path = argv[1];
    const auto num_levels_opt = ParseNumLevels(argc, argv, 6, 2);
    if (!num_levels_opt.has_value()) {
        PrintNumLevelsParseError(argc, argv, 2);
        return EXIT_FAILURE;
    }
    const int num_levels = *num_levels_opt;

    cv::Mat input;
    bool is_color = false;
    if (!LoadFloat01(image_path, input, &is_color)) {
        std::cerr << "Error: could not read image: " << image_path << "\n";
        return EXIT_FAILURE;
    }

    if (is_color) {
        if (!RunColorDemo(input, num_levels, out_dir)) {
            return EXIT_FAILURE;
        }
    } else {
        if (!RunGrayscaleDemo(input, num_levels, out_dir)) {
            return EXIT_FAILURE;
        }
    }

    return EXIT_SUCCESS;
}
