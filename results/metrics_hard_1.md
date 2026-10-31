| Tracker | mean error (px) | median (px) | max (px) | success IoU>=0.5 (%) | lost frames (>20 px) | mean error visible (px) | mean error hidden (px) | ms/frame |
|---|---|---|---|---|---|---|---|---|
| LK + Kalman + template (this repo) | 1.09 | 1.24 | 2.53 | 100.00 | 0 | 1.02 | 2.14 | 1.27 |
| Plain LK (median flow) | 39.68 | 35.42 | 123.14 | 40.47 | 175 | 40.36 | 30.25 | 1.06 |
| OpenCV MIL | 53.17 | 29.05 | 161.71 | 39.80 | 180 | 51.25 | 79.97 | 69.76 |
