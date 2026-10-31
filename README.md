# Video Object Tracking (C++ and Python, OpenCV)

Author: BEN, Software Engineer · 2026-11-01

Single-object tracking in video that survives a full occlusion. The tracker combines Lucas-Kanade
optical flow, a constant-velocity Kalman filter and template re-acquisition, and is compared with two baselines.

## What is in here
- `cpp/make_synthetic_video.cpp` – makes a test video with a known path (ground truth CSV). Scenarios: `clean`, and `hard` (noise + the object is hidden behind a bar for frames 120-139).
- `cpp/track_video.cpp` – the three trackers, the metrics and the picture output (C++).
- `python/track_video.py` – the same tracker, metrics and CSV output in Python (reads the video made by the C++ generator).
- `tools/plot_results.py` – error-vs-frame plot.

## Build and run
```bash
sudo apt install libopencv-dev cmake g++
cmake -S . -B build && cmake --build build
./build/make_synthetic_video --out hard.avi --gt hard.csv --scenario hard --seed 1
./build/track_video --video hard.avi --gt hard.csv --out results --tag hard_1
pip install numpy opencv-python matplotlib
python3 python/track_video.py --video hard.avi --gt hard.csv --tag hard_1   # Python version
python3 tools/plot_results.py hard_1 results/error_hard_1.png
```
Your own video: `--init cx,cy,w,h` (box in the first frame) instead of `--gt`.

## Measured results (OpenCV 4.6, 300 frames, 320x240, 3 seeds each)
Success = box IoU >= 0.5. Lost = centre error > 20 px.

| Scenario | Tracker | mean error (px) | success (%) | lost frames | ms/frame |
|---|---|---|---|---|---|
| clean | LK + Kalman + template | 1.00-1.03 | 100 | 0 | 1.2-1.5 |
| clean | Plain LK | 0.05-0.08 | 100 | 0 | 0.8-0.9 |
| clean | OpenCV MIL | 0.79-2.04 | 100 | 0 | 65-70 |
| hard | LK + Kalman + template | 1.07-1.09 | 100 | 0 | 1.1-1.6 |
| hard | Plain LK | 30.2-39.7 | 40-41 | 173-175 | ~1.0 |
| hard | OpenCV MIL | 45.6-53.2 | ~40 | 179-180 | 69-71 |

Per-run tables kept: `results/metrics_clean_1.md` and `metrics_hard_1..3.md` (the other clean seeds are re-created with the commands above). Picture: `results/montage_hard_1.png`
(green = ground truth, red = this tracker, blue = plain LK, yellow = MIL).

Python and C++ give the same baseline numbers; the Kalman tracker is close but not identical (for example clean mean error 0.9 px in Python vs 1.0 px in C++, hard 1.05 vs 1.09 px) because of small differences in feature selection.

## Honest limits
- The video is synthetic, with a textured object on a textured background. Real video will be harder.
- On the clean video, plain LK is more accurate (about 0.06 px vs 1.0 px). The Kalman filter adds smoothing lag; its benefit shows when the object is hidden.
- Kalman noise values were tuned on these videos, so the numbers are optimistic for other footage.
- One object, one camera, no scale or rotation change.

## License
MIT, see `LICENSE`.
