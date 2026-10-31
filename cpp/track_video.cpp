// BEN 2026-11-01
// Change aim: Track one object in a video with optical flow + Kalman filter and compare with two baselines.
// File: track_video.cpp
#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/core/utils/filesystem.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

struct GtRow { int frame; double cx, cy, w, h; int visible; };
struct TrackRow { double cx, cy, w, h; int state; };   // state 1 = tracking, 0 = coasting (object lost)

// BEN 2026-11-01
// Change aim: Parse "--name value" options with defaults.
// File: track_video.cpp
// Function: getArg
static std::string getArg(int argc, char** argv, const std::string& name, const std::string& def)
{
    for (int i = 1; i + 1 < argc; ++i)
        if (name == argv[i]) return argv[i + 1];
    return def;
}

// BEN 2026-11-01
// Change aim: Read the ground-truth CSV written by make_synthetic_video.
// File: track_video.cpp
// Function: readGt
static std::vector<GtRow> readGt(const std::string& path)
{
    std::vector<GtRow> rows;
    std::ifstream in(path);
    std::string line;
    std::getline(in, line);   // header
    while (std::getline(in, line)) {
        std::stringstream ss(line);
        std::string tok;
        GtRow r;
        std::getline(ss, tok, ','); r.frame = std::stoi(tok);
        std::getline(ss, tok, ','); r.cx = std::stod(tok);
        std::getline(ss, tok, ','); r.cy = std::stod(tok);
        std::getline(ss, tok, ','); r.w = std::stod(tok);
        std::getline(ss, tok, ','); r.h = std::stod(tok);
        std::getline(ss, tok, ','); r.visible = std::stoi(tok);
        rows.push_back(r);
    }
    return rows;
}

// BEN 2026-11-01
// Change aim: Median of a list of numbers (robust to the points that follow something else).
// File: track_video.cpp
// Function: medianOf
static double medianOf(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// BEN 2026-11-01
// Change aim: Find corner features inside the middle 70% of a box.
// File: track_video.cpp
// Function: detectFeatures
static void detectFeatures(const cv::Mat& gray, const cv::Rect& box, std::vector<cv::Point2f>& pts)
{
    cv::Rect inner(box.x + box.width * 15 / 100, box.y + box.height * 15 / 100,
                   box.width * 70 / 100, box.height * 70 / 100);
    inner &= cv::Rect(0, 0, gray.cols, gray.rows);
    pts.clear();
    if (inner.width < 8 || inner.height < 8) return;
    cv::Mat mask = cv::Mat::zeros(gray.size(), CV_8U);
    mask(inner).setTo(255);
    cv::goodFeaturesToTrack(gray, pts, 80, 0.01, 4, mask);
}

struct FlowResult {
    std::vector<cv::Point2f> next;   // inlier points in the new frame
    cv::Point2f median;              // median displacement of the inliers
    bool ok;
};

// BEN 2026-11-01
// Change aim: Lucas-Kanade flow with a forward-backward check, then keep points that agree with the median.
// File: track_video.cpp
// Function: trackPoints
static FlowResult trackPoints(const cv::Mat& prevGray, const cv::Mat& gray, const std::vector<cv::Point2f>& prevPts)
{
    FlowResult res;
    res.ok = false;
    res.median = cv::Point2f(0, 0);
    if (prevPts.size() < 4) return res;

    std::vector<cv::Point2f> fwd, back;
    std::vector<uchar> st1, st2;
    std::vector<float> e1, e2;
    const cv::Size win(21, 21);
    const cv::TermCriteria crit(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);
    cv::calcOpticalFlowPyrLK(prevGray, gray, prevPts, fwd, st1, e1, win, 3, crit);
    cv::calcOpticalFlowPyrLK(gray, prevGray, fwd, back, st2, e2, win, 3, crit);

    std::vector<cv::Point2f> a, b;
    std::vector<double> dx, dy;
    for (size_t i = 0; i < prevPts.size(); ++i) {
        if (!st1[i] || !st2[i]) continue;
        if (cv::norm(prevPts[i] - back[i]) > 1.0) continue;          // forward-backward error
        a.push_back(prevPts[i]);
        b.push_back(fwd[i]);
        dx.push_back(fwd[i].x - prevPts[i].x);
        dy.push_back(fwd[i].y - prevPts[i].y);
    }
    if (a.size() < 4) return res;
    const cv::Point2f med(static_cast<float>(medianOf(dx)), static_cast<float>(medianOf(dy)));
    for (size_t i = 0; i < a.size(); ++i)
        if (cv::norm(cv::Point2f(static_cast<float>(dx[i]), static_cast<float>(dy[i])) - med) < 2.0)
            res.next.push_back(b[i]);
    res.median = med;
    res.ok = res.next.size() >= 4;
    return res;
}

// BEN 2026-11-01
// Change aim: Template of the first-frame object; scores how well a place in a later frame matches it.
// File: track_video.cpp
// Function: TemplateMatcher
class TemplateMatcher {
public:
    // BEN 2026-11-01
    // Change aim: Cut the template from the first frame and remember the sub-pixel offset of the cut.
    // File: track_video.cpp
    // Function: TemplateMatcher::init
    void init(const cv::Mat& gray, double cx, double cy, int w, int h)
    {
        const int x = static_cast<int>(std::lround(cx - w / 2.0));
        const int y = static_cast<int>(std::lround(cy - h / 2.0));
        templ_ = gray(cv::Rect(x, y, w, h)).clone();
        biasX_ = cx - (x + w / 2.0);
        biasY_ = cy - (y + h / 2.0);
    }

    // BEN 2026-11-01
    // Change aim: Best normalised correlation within +/- radius pixels of a centre; returns the best centre.
    // File: track_video.cpp
    // Function: TemplateMatcher::search
    double search(const cv::Mat& gray, const cv::Point2f& centre, int radius, cv::Point2f& best) const
    {
        const int tw = templ_.cols, th = templ_.rows;
        cv::Rect r(static_cast<int>(std::lround(centre.x - tw / 2.0)) - radius,
                   static_cast<int>(std::lround(centre.y - th / 2.0)) - radius, tw + 2 * radius, th + 2 * radius);
        r &= cv::Rect(0, 0, gray.cols, gray.rows);
        if (r.width < tw || r.height < th) return -1.0;
        cv::Mat res;
        cv::matchTemplate(gray(r), templ_, res, cv::TM_CCOEFF_NORMED);
        double mx;
        cv::Point loc;
        cv::minMaxLoc(res, nullptr, &mx, nullptr, &loc);
        best = cv::Point2f(static_cast<float>(r.x + loc.x + tw / 2.0 + biasX_),
                           static_cast<float>(r.y + loc.y + th / 2.0 + biasY_));
        return mx;
    }

private:
    cv::Mat templ_;
    double biasX_ = 0, biasY_ = 0;
};

// BEN 2026-11-01
// Change aim: Tracker 1: LK optical flow + constant-velocity Kalman filter + template re-acquisition.
// File: track_video.cpp
// Function: LkKalmanTracker
class LkKalmanTracker {
public:
    // BEN 2026-11-01
    // Change aim: Start the tracker on the first frame with a known box centre and size.
    // File: track_video.cpp
    // Function: LkKalmanTracker::init
    void init(const cv::Mat& gray, double cx, double cy, int w, int h)
    {
        w_ = w;
        h_ = h;
        kf_ = cv::KalmanFilter(4, 2, 0, CV_32F);
        kf_.transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1);
        kf_.measurementMatrix = (cv::Mat_<float>(2, 4) << 1, 0, 0, 0, 0, 1, 0, 0);
        cv::setIdentity(kf_.processNoiseCov, cv::Scalar::all(1e-2));
        kf_.processNoiseCov.at<float>(2, 2) = 1e-1f;
        kf_.processNoiseCov.at<float>(3, 3) = 1e-1f;
        cv::setIdentity(kf_.measurementNoiseCov, cv::Scalar::all(0.05));
        cv::setIdentity(kf_.errorCovPost, cv::Scalar::all(1.0));
        kf_.errorCovPost.at<float>(2, 2) = 25.f;
        kf_.errorCovPost.at<float>(3, 3) = 25.f;
        kf_.statePost = (cv::Mat_<float>(4, 1) << static_cast<float>(cx), static_cast<float>(cy), 0.f, 0.f);
        est_ = cv::Point2f(static_cast<float>(cx), static_cast<float>(cy));
        tm_.init(gray, cx, cy, w, h);
        reseed(gray);
        prevGray_ = gray.clone();
        tracking_ = true;
    }

    // BEN 2026-11-01
    // Change aim: Process one frame: predict, measure with optical flow, check the look, correct or coast.
    // File: track_video.cpp
    // Function: LkKalmanTracker::update
    TrackRow update(const cv::Mat& gray)
    {
        const cv::Mat pred = kf_.predict();
        const cv::Point2f predC(pred.at<float>(0), pred.at<float>(1));
        const cv::Point2f predV(pred.at<float>(2), pred.at<float>(3));

        bool measured = false;
        cv::Point2f z;
        std::vector<cv::Point2f> nextPts;
        if (tracking_) {
            const FlowResult fr = trackPoints(prevGray_, gray, pts_);
            if (fr.ok && cv::norm(fr.median - predV) < 8.0) {     // flow must agree with the motion model
                z = est_ + fr.median;
                cv::Point2f best;
                if (tm_.search(gray, z, 1, best) > 0.35) {         // and the object must still look the same
                    measured = true;
                    nextPts = fr.next;
                }
            }
        }

        if (measured) {
            kf_.correct((cv::Mat_<float>(2, 1) << z.x, z.y));
            tracking_ = true;
            pts_ = nextPts;
            if (pts_.size() < 25) reseed(gray);
        } else {
            kf_.statePre.copyTo(kf_.statePost);               // no measurement: keep the prediction
            kf_.errorCovPre.copyTo(kf_.errorCovPost);
            tracking_ = false;
            pts_.clear();
            cv::Point2f best;
            if (tm_.search(gray, predC, 25, best) > 0.6) {     // try to find the object again
                kf_.correct((cv::Mat_<float>(2, 1) << best.x, best.y));
                tracking_ = true;
                est_ = cv::Point2f(kf_.statePost.at<float>(0), kf_.statePost.at<float>(1));
                reseed(gray);
            }
        }
        est_ = cv::Point2f(kf_.statePost.at<float>(0), kf_.statePost.at<float>(1));
        prevGray_ = gray.clone();
        return {est_.x, est_.y, static_cast<double>(w_), static_cast<double>(h_), tracking_ ? 1 : 0};
    }

private:
    // BEN 2026-11-01
    // Change aim: Pick new features inside the current box estimate.
    // File: track_video.cpp
    // Function: LkKalmanTracker::reseed
    void reseed(const cv::Mat& gray)
    {
        detectFeatures(gray, cv::Rect(static_cast<int>(std::lround(est_.x - w_ / 2.0)),
                                      static_cast<int>(std::lround(est_.y - h_ / 2.0)), w_, h_), pts_);
    }

    cv::KalmanFilter kf_;
    TemplateMatcher tm_;
    cv::Mat prevGray_;
    std::vector<cv::Point2f> pts_;
    cv::Point2f est_;
    int w_ = 0, h_ = 0;
    bool tracking_ = false;
};

// BEN 2026-11-01
// Change aim: Baseline 1: plain LK optical flow (median shift), no Kalman filter, no re-acquisition.
// File: track_video.cpp
// Function: PlainLkTracker
class PlainLkTracker {
public:
    void init(const cv::Mat& gray, double cx, double cy, int w, int h)
    {
        c_ = cv::Point2f(static_cast<float>(cx), static_cast<float>(cy));
        w_ = w;
        h_ = h;
        reseed(gray);
        prevGray_ = gray.clone();
    }

    // BEN 2026-11-01
    // Change aim: Move the box by the median flow; if the flow fails, stay in place and pick new features.
    // File: track_video.cpp
    // Function: PlainLkTracker::update
    TrackRow update(const cv::Mat& gray)
    {
        const FlowResult fr = trackPoints(prevGray_, gray, pts_);
        int state = 0;
        if (fr.ok) {
            c_ += fr.median;
            pts_ = fr.next;
            state = 1;
        } else {
            pts_.clear();
        }
        if (pts_.size() < 15) reseed(gray);
        prevGray_ = gray.clone();
        return {c_.x, c_.y, static_cast<double>(w_), static_cast<double>(h_), state};
    }

private:
    void reseed(const cv::Mat& gray)
    {
        detectFeatures(gray, cv::Rect(static_cast<int>(std::lround(c_.x - w_ / 2.0)),
                                      static_cast<int>(std::lround(c_.y - h_ / 2.0)), w_, h_), pts_);
    }
    cv::Point2f c_;
    int w_ = 0, h_ = 0;
    cv::Mat prevGray_;
    std::vector<cv::Point2f> pts_;
};

// BEN 2026-11-01
// Change aim: Baseline 2: OpenCV's built-in MIL tracker, wrapped to the same interface.
// File: track_video.cpp
// Function: MilTracker
class MilTracker {
public:
    void init(const cv::Mat& frame, double cx, double cy, int w, int h)
    {
        tracker_ = cv::TrackerMIL::create();
        last_ = cv::Rect2d(cx - w / 2.0, cy - h / 2.0, w, h);
        tracker_->init(frame, last_);
    }
    TrackRow update(const cv::Mat& frame)
    {
        cv::Rect box;
        const bool ok = tracker_->update(frame, box);
        if (ok) last_ = cv::Rect2d(box);
        return {last_.x + last_.width / 2.0, last_.y + last_.height / 2.0, last_.width, last_.height, ok ? 1 : 0};
    }

private:
    cv::Ptr<cv::TrackerMIL> tracker_;
    cv::Rect2d last_;
};

struct Metrics {
    double meanErr = 0, medErr = 0, maxErr = 0, success = 0, meanVisible = 0, meanHidden = 0, msPerFrame = 0;
    int lost = 0;
};

// BEN 2026-11-01
// Change aim: Intersection over union of two axis-aligned boxes given by centre and size.
// File: track_video.cpp
// Function: iou
static double iou(double ax, double ay, double aw, double ah, double bx, double by, double bw, double bh)
{
    const double x1 = std::max(ax - aw / 2, bx - bw / 2), x2 = std::min(ax + aw / 2, bx + bw / 2);
    const double y1 = std::max(ay - ah / 2, by - bh / 2), y2 = std::min(ay + ah / 2, by + bh / 2);
    const double inter = std::max(0.0, x2 - x1) * std::max(0.0, y2 - y1);
    const double uni = aw * ah + bw * bh - inter;
    return uni > 0 ? inter / uni : 0.0;
}

// BEN 2026-11-01
// Change aim: Compare one tracker's output with the ground truth (frame 0 is the start and is skipped).
// File: track_video.cpp
// Function: evaluate
static Metrics evaluate(const std::vector<TrackRow>& tr, const std::vector<GtRow>& gt, double totalMs)
{
    Metrics m;
    std::vector<double> errs;
    double sumVis = 0, sumHid = 0;
    int nVis = 0, nHid = 0, ok = 0;
    for (size_t i = 1; i < tr.size() && i < gt.size(); ++i) {
        const double e = std::hypot(tr[i].cx - gt[i].cx, tr[i].cy - gt[i].cy);
        errs.push_back(e);
        if (e > 20.0) ++m.lost;
        if (iou(tr[i].cx, tr[i].cy, tr[i].w, tr[i].h, gt[i].cx, gt[i].cy, gt[i].w, gt[i].h) >= 0.5) ++ok;
        if (gt[i].visible) { sumVis += e; ++nVis; } else { sumHid += e; ++nHid; }
    }
    double s = 0;
    for (double e : errs) { s += e; m.maxErr = std::max(m.maxErr, e); }
    m.meanErr = s / errs.size();
    m.medErr = medianOf(errs);
    m.success = 100.0 * ok / errs.size();
    m.meanVisible = nVis ? sumVis / nVis : 0.0;
    m.meanHidden = nHid ? sumHid / nHid : 0.0;
    m.msPerFrame = totalMs / tr.size();
    return m;
}

// BEN 2026-11-01
// Change aim: Write one tracker's per-frame result to a CSV file.
// File: track_video.cpp
// Function: writeTrack
static void writeTrack(const std::string& path, const std::vector<TrackRow>& tr)
{
    std::ofstream out(path);
    out << "frame,cx,cy,w,h,state\n";
    for (size_t i = 0; i < tr.size(); ++i)
        out << i << "," << tr[i].cx << "," << tr[i].cy << "," << tr[i].w << "," << tr[i].h << "," << tr[i].state << "\n";
}

// BEN 2026-11-01
// Change aim: Draw one centred box and a tiny label on a frame.
// File: track_video.cpp
// Function: drawBox
static void drawBox(cv::Mat& img, double cx, double cy, double w, double h, const cv::Scalar& colour, int thickness)
{
    cv::rectangle(img, cv::Rect(static_cast<int>(std::lround(cx - w / 2)), static_cast<int>(std::lround(cy - h / 2)),
                                static_cast<int>(w), static_cast<int>(h)), colour, thickness);
}

// BEN 2026-11-01
// Change aim: Run the three trackers on the video, print the comparison table and save CSV and picture files.
// File: track_video.cpp
// Function: main
int main(int argc, char** argv)
{
    const std::string videoPath = getArg(argc, argv, "--video", "");
    const std::string gtPath = getArg(argc, argv, "--gt", "");
    const std::string outDir = getArg(argc, argv, "--out", "results");
    const std::string tag = getArg(argc, argv, "--tag", "run");
    const std::string initArg = getArg(argc, argv, "--init", "");   // "cx,cy,w,h" when there is no ground truth
    if (videoPath.empty()) {
        std::cerr << "usage: track_video --video file.avi [--gt gt.csv | --init cx,cy,w,h] [--out dir] [--tag name]\n";
        return 1;
    }

    std::vector<GtRow> gt;
    if (!gtPath.empty()) gt = readGt(gtPath);
    double cx0 = 0, cy0 = 0, w0 = 0, h0 = 0;
    if (!gt.empty()) {
        cx0 = gt[0].cx; cy0 = gt[0].cy; w0 = gt[0].w; h0 = gt[0].h;
    } else if (!initArg.empty()) {
        char c;
        std::stringstream ss(initArg);
        ss >> cx0 >> c >> cy0 >> c >> w0 >> c >> h0;
    } else {
        std::cerr << "give --gt or --init\n";
        return 1;
    }

    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "cannot open " << videoPath << "\n";
        return 1;
    }

    LkKalmanTracker lkk;
    PlainLkTracker plain;
    MilTracker mil;
    std::vector<TrackRow> trK, trP, trM;
    double msK = 0, msP = 0, msM = 0;
    std::vector<cv::Mat> frames;

    cv::Mat frame, gray;
    int index = 0;
    while (cap.read(frame)) {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        frames.push_back(frame.clone());
        typedef std::chrono::steady_clock Clock;
        if (index == 0) {
            lkk.init(gray, cx0, cy0, static_cast<int>(w0), static_cast<int>(h0));
            plain.init(gray, cx0, cy0, static_cast<int>(w0), static_cast<int>(h0));
            mil.init(frame, cx0, cy0, static_cast<int>(w0), static_cast<int>(h0));
            trK.push_back({cx0, cy0, w0, h0, 1});
            trP.push_back({cx0, cy0, w0, h0, 1});
            trM.push_back({cx0, cy0, w0, h0, 1});
        } else {
            Clock::time_point t0 = Clock::now();
            trK.push_back(lkk.update(gray));
            Clock::time_point t1 = Clock::now();
            trP.push_back(plain.update(gray));
            Clock::time_point t2 = Clock::now();
            trM.push_back(mil.update(frame));
            Clock::time_point t3 = Clock::now();
            msK += std::chrono::duration<double, std::milli>(t1 - t0).count();
            msP += std::chrono::duration<double, std::milli>(t2 - t1).count();
            msM += std::chrono::duration<double, std::milli>(t3 - t2).count();
        }
        ++index;
    }
    std::cout << "frames: " << index << "\n";

    cv::utils::fs::createDirectories(outDir);
    writeTrack(outDir + "/track_lk_kalman_" + tag + ".csv", trK);
    writeTrack(outDir + "/track_plain_lk_" + tag + ".csv", trP);
    writeTrack(outDir + "/track_mil_" + tag + ".csv", trM);

    if (!gt.empty()) {
        const Metrics mk = evaluate(trK, gt, msK), mp = evaluate(trP, gt, msP), mm = evaluate(trM, gt, msM);
        std::ostringstream t;
        t << std::fixed << std::setprecision(2);
        t << "| Tracker | mean error (px) | median (px) | max (px) | success IoU>=0.5 (%) | lost frames (>20 px) | "
             "mean error visible (px) | mean error hidden (px) | ms/frame |\n";
        t << "|---|---|---|---|---|---|---|---|---|\n";
        const char* names[3] = {"LK + Kalman + template (this repo)", "Plain LK (median flow)", "OpenCV MIL"};
        const Metrics* ms[3] = {&mk, &mp, &mm};
        for (int i = 0; i < 3; ++i)
            t << "| " << names[i] << " | " << ms[i]->meanErr << " | " << ms[i]->medErr << " | " << ms[i]->maxErr << " | "
              << ms[i]->success << " | " << ms[i]->lost << " | " << ms[i]->meanVisible << " | " << ms[i]->meanHidden
              << " | " << ms[i]->msPerFrame << " |\n";
        std::cout << t.str();
        std::ofstream(outDir + "/metrics_" + tag + ".md") << t.str();
    }

    // picture: a few frames with the ground truth (green), LK+Kalman (red), plain LK (blue) and MIL (yellow)
    const int picks[6] = {1, 60, 125, 135, 150, 250};
    std::vector<cv::Mat> tiles;
    for (int k = 0; k < 6; ++k) {
        const int f = std::min(picks[k], static_cast<int>(frames.size()) - 1);
        cv::Mat img = frames[f].clone();
        if (!gt.empty() && f < static_cast<int>(gt.size())) drawBox(img, gt[f].cx, gt[f].cy, gt[f].w + 8, gt[f].h + 8, cv::Scalar(0, 255, 0), 1);
        drawBox(img, trP[f].cx, trP[f].cy, trP[f].w, trP[f].h, cv::Scalar(255, 120, 0), 1);
        drawBox(img, trM[f].cx, trM[f].cy, trM[f].w, trM[f].h, cv::Scalar(0, 255, 255), 1);
        drawBox(img, trK[f].cx, trK[f].cy, trK[f].w, trK[f].h, cv::Scalar(0, 0, 255), 2);
        cv::putText(img, "frame " + std::to_string(f), cv::Point(6, 16), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        tiles.push_back(img);
    }
    cv::Mat row1, row2, montage;
    cv::hconcat(std::vector<cv::Mat>(tiles.begin(), tiles.begin() + 3), row1);
    cv::hconcat(std::vector<cv::Mat>(tiles.begin() + 3, tiles.end()), row2);
    cv::vconcat(row1, row2, montage);
    cv::imwrite(outDir + "/montage_" + tag + ".png", montage);
    std::cout << "saved CSV files and montage_" << tag << ".png in " << outDir << "\n";
    return 0;
}
