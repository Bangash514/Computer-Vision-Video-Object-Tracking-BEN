| Tracker | mean error (px) | median (px) | max (px) | success IoU>=0.5 (%) | lost frames (>20 px) | mean error visible (px) | mean error hidden (px) | ms/frame |
|---|---|---|---|---|---|---|---|---|
| LK + Kalman + template (this repo) | 1.07 | 1.23 | 2.05 | 100.00 | 0 | 1.02 | 1.76 | 1.14 |
| Plain LK (median flow) | 30.20 | 27.12 | 104.77 | 41.47 | 173 | 30.22 | 29.93 | 1.02 |
| OpenCV MIL | 45.57 | 32.36 | 137.92 | 39.80 | 179 | 44.67 | 58.15 | 71.26 |
