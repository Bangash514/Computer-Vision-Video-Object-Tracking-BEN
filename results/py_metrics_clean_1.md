| Tracker | mean error (px) | median (px) | max (px) | success IoU>=0.5 (%) | lost frames (>20 px) | mean error visible (px) | mean error hidden (px) | ms/frame |
|---|---|---|---|---|---|---|---|---|
| LK + Kalman + template (this repo) | 0.88 | 0.95 | 1.53 | 100.00 | 0 | 0.88 | 0.00 | 1.77 |
| Plain LK (median flow) | 0.08 | 0.08 | 0.17 | 100.00 | 0 | 0.08 | 0.00 | 1.05 |
| OpenCV MIL | 2.04 | 2.03 | 4.38 | 100.00 | 0 | 2.04 | 0.00 | 82.96 |
