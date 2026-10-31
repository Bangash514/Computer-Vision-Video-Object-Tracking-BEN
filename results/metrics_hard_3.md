| Tracker | mean error (px) | median (px) | max (px) | success IoU>=0.5 (%) | lost frames (>20 px) | mean error visible (px) | mean error hidden (px) | ms/frame |
|---|---|---|---|---|---|---|---|---|
| LK + Kalman + template (this repo) | 1.09 | 1.21 | 2.36 | 100.00 | 0 | 1.02 | 2.04 | 1.57 |
| Plain LK (median flow) | 31.45 | 29.22 | 106.50 | 40.80 | 174 | 31.40 | 32.14 | 1.00 |
| OpenCV MIL | 46.75 | 33.92 | 135.97 | 39.80 | 180 | 46.18 | 54.71 | 68.77 |
