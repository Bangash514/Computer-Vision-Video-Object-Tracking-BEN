// BEN 2026-11-01
// Change aim: Make a synthetic test video with a known object path (ground truth) for tracker tests.
// File: make_synthetic_video.cpp
#include <opencv2/opencv.hpp>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

static const double kPi = 3.14159265358979323846;

// BEN 2026-11-01
// Change aim: Parse "--name value" options with defaults.
// File: make_synthetic_video.cpp
// Function: getArg
static std::string getArg(int argc, char** argv, const std::string& name, const std::string& def)
{
    for (int i = 1; i + 1 < argc; ++i)
        if (name == argv[i]) return argv[i + 1];
    return def;
}

// BEN 2026-11-01
// Change aim: Build a smooth random texture in the range [lo, hi] (used for the background and the object).
// File: make_synthetic_video.cpp
// Function: makeTexture
static cv::Mat makeTexture(cv::RNG& rng, int rows, int cols, double sigma, double lo, double hi)
{
    cv::Mat t(rows, cols, CV_32F);
    rng.fill(t, cv::RNG::UNIFORM, 0.0, 1.0);
    cv::GaussianBlur(t, t, cv::Size(0, 0), sigma);
    cv::normalize(t, t, lo, hi, cv::NORM_MINMAX);
    return t;
}

// BEN 2026-11-01
// Change aim: Render the frames, write the video and the ground-truth CSV.
// File: make_synthetic_video.cpp
// Function: main
int main(int argc, char** argv)
{
    const std::string outVideo = getArg(argc, argv, "--out", "synthetic.avi");
    const std::string outGt = getArg(argc, argv, "--gt", "synthetic_gt.csv");
    const std::string scenario = getArg(argc, argv, "--scenario", "hard");   // clean | hard
    const int frames = std::stoi(getArg(argc, argv, "--frames", "300"));
    const int seed = std::stoi(getArg(argc, argv, "--seed", "1"));
    const bool hard = (scenario == "hard");

    const int W = 320, H = 240, OBJ = 40;
    cv::RNG rng(static_cast<uint64>(seed));
    cv::RNG noiseRng(static_cast<uint64>(seed) + 1000);

    const cv::Mat bg = makeTexture(rng, H, W, 3.0, 70.0, 150.0);

    // object: high-contrast texture, dark border and a bright cross so it has corners to track
    cv::Mat patch = makeTexture(rng, OBJ, OBJ, 1.2, 40.0, 230.0);
    cv::rectangle(patch, cv::Rect(0, 0, OBJ, OBJ), cv::Scalar(20), 3);
    cv::line(patch, cv::Point(8, 8), cv::Point(OBJ - 9, OBJ - 9), cv::Scalar(250), 2);
    cv::line(patch, cv::Point(OBJ - 9, 8), cv::Point(8, OBJ - 9), cv::Scalar(10), 2);
    const cv::Mat mask = cv::Mat::ones(OBJ, OBJ, CV_32F);

    cv::VideoWriter writer(outVideo, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30.0, cv::Size(W, H));
    if (!writer.isOpened()) {
        std::cerr << "Cannot open video for writing: " << outVideo << std::endl;
        return 1;
    }
    std::ofstream gt(outGt);
    gt << "frame,cx,cy,w,h,visible\n";

    const int occStart = 120, occEnd = 140;      // hard scenario: object fully hidden in this range
    for (int i = 0; i < frames; ++i) {
        const double cx = 160.0 + 90.0 * std::sin(2 * kPi * i / 240.0);
        const double cy = 120.0 + 60.0 * std::sin(2 * kPi * i / 150.0 + 1.0);

        const cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, cx - OBJ / 2.0, 0, 1, cy - OBJ / 2.0);
        cv::Mat patchBig, maskBig;
        cv::warpAffine(patch, patchBig, M, cv::Size(W, H), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
        cv::warpAffine(mask, maskBig, M, cv::Size(W, H), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
        cv::Mat frame = bg.mul(1.0 - maskBig) + patchBig.mul(maskBig);

        bool visible = true;
        if (hard && i >= occStart && i < occEnd) {          // flat grey bar hides the object
            const int x0 = static_cast<int>(std::round(cx)) - 35;
            cv::rectangle(frame, cv::Rect(x0, 0, 70, H), cv::Scalar(100), cv::FILLED);
            visible = false;
        }
        if (hard) {
            cv::Mat n(H, W, CV_32F);
            noiseRng.fill(n, cv::RNG::NORMAL, 0.0, 6.0);     // sensor noise, sigma 6 grey levels
            frame += n;
        }
        cv::Mat f8, bgr;
        frame.convertTo(f8, CV_8U);
        cv::cvtColor(f8, bgr, cv::COLOR_GRAY2BGR);
        writer << bgr;
        gt << i << "," << cx << "," << cy << "," << OBJ << "," << OBJ << "," << (visible ? 1 : 0) << "\n";
    }
    std::cout << "wrote " << frames << " frames (" << scenario << ", seed " << seed << ") to " << outVideo
              << " and " << outGt << std::endl;
    return 0;
}
