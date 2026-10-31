# BEN 2026-11-01
# Change aim: Python port of track_video.cpp: LK optical flow + Kalman + template re-acquisition vs two baselines.
# File: track_video.py
import argparse
import csv
import time
import numpy as np
import cv2

LK = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 0.01))


# BEN 2026-11-01
# Change aim: Corner features inside the middle 70% of a box.
# File: track_video.py
# Function: detect_features
def detect_features(gray, c, w, h):
    x, y = int(round(c[0] - w / 2)), int(round(c[1] - h / 2))
    ix, iy, iw, ih = x + w * 15 // 100, y + h * 15 // 100, w * 70 // 100, h * 70 // 100
    ix0, iy0 = max(ix, 0), max(iy, 0)
    ix1, iy1 = min(ix + iw, gray.shape[1]), min(iy + ih, gray.shape[0])
    if ix1 - ix0 < 8 or iy1 - iy0 < 8:
        return np.zeros((0, 1, 2), np.float32)
    mask = np.zeros(gray.shape, np.uint8)
    mask[iy0:iy1, ix0:ix1] = 255
    p = cv2.goodFeaturesToTrack(gray, 80, 0.01, 4, mask=mask)
    return p if p is not None else np.zeros((0, 1, 2), np.float32)


# BEN 2026-11-01
# Change aim: LK flow with forward-backward check; keep points that agree with the median displacement.
# File: track_video.py
# Function: track_points
def track_points(prev, gray, pts):
    if len(pts) < 4:
        return None, None
    fwd, s1, _ = cv2.calcOpticalFlowPyrLK(prev, gray, pts, None, **LK)
    back, s2, _ = cv2.calcOpticalFlowPyrLK(gray, prev, fwd, None, **LK)
    fb = np.linalg.norm((pts - back).reshape(-1, 2), axis=1)
    good = (s1.ravel() == 1) & (s2.ravel() == 1) & (fb <= 1.0)
    if good.sum() < 4:
        return None, None
    d = (fwd - pts).reshape(-1, 2)[good]
    med = np.median(d, axis=0)
    keep = np.linalg.norm(d - med, axis=1) < 2.0
    nxt = fwd[good][keep]
    if len(nxt) < 4:
        return None, None
    return med, nxt


# BEN 2026-11-01
# Change aim: Template of the first-frame object; normalised correlation search near a centre.
# File: track_video.py
# Function: Template
class Template:
    def __init__(self, gray, cx, cy, w, h):
        self.w, self.h = w, h
        x, y = int(round(cx - w / 2)), int(round(cy - h / 2))
        self.t = gray[y:y + h, x:x + w].copy()
        self.bx, self.by = cx - (x + w / 2), cy - (y + h / 2)

    def search(self, gray, c, radius):
        x = int(round(c[0] - self.w / 2)) - radius
        y = int(round(c[1] - self.h / 2)) - radius
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + self.w + 2 * radius, gray.shape[1]), min(y + self.h + 2 * radius, gray.shape[0])
        if x1 - x0 < self.w or y1 - y0 < self.h:
            return None, -1.0
        res = cv2.matchTemplate(gray[y0:y1, x0:x1], self.t, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(res)
        return np.array([x0 + loc[0] + self.w / 2 + self.bx, y0 + loc[1] + self.h / 2 + self.by]), mx


# BEN 2026-11-01
# Change aim: Tracker 1: LK + Kalman + template check and re-acquisition.
# File: track_video.py
# Function: LkKalmanTracker
class LkKalmanTracker:
    def __init__(self, gray, cx, cy, w, h):
        self.w, self.h = w, h
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kf.processNoiseCov = np.diag([0.01, 0.01, 0.1, 0.1]).astype(np.float32)
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.05
        kf.errorCovPost = np.diag([1, 1, 25, 25]).astype(np.float32)
        kf.statePost = np.array([[cx], [cy], [0], [0]], np.float32)
        self.kf, self.tm = kf, Template(gray, cx, cy, w, h)
        self.est = np.array([cx, cy], np.float32)
        self.prev, self.tracking = gray.copy(), True
        self.pts = detect_features(gray, self.est, w, h)

    def update(self, gray):
        p = self.kf.predict()
        pc, pv = p[:2, 0], p[2:, 0]
        measured, z, nxt = False, None, None
        if self.tracking:
            med, nxt = track_points(self.prev, gray, self.pts)
            if med is not None and np.linalg.norm(med - pv) < 8.0:
                z = self.est + med
                _, sc = self.tm.search(gray, z, 1)
                measured = sc > 0.35
        if measured:
            self.kf.correct(z.reshape(2, 1).astype(np.float32))
            self.tracking, self.pts = True, nxt
            if len(self.pts) < 25:
                self.pts = detect_features(gray, self.kf.statePost[:2, 0], self.w, self.h)
        else:
            self.kf.statePost = self.kf.statePre.copy()
            self.kf.errorCovPost = self.kf.errorCovPre.copy()
            self.tracking, self.pts = False, np.zeros((0, 1, 2), np.float32)
            best, sc = self.tm.search(gray, pc, 25)
            if best is not None and sc > 0.6:
                self.kf.correct(best.reshape(2, 1).astype(np.float32))
                self.tracking = True
                self.pts = detect_features(gray, self.kf.statePost[:2, 0], self.w, self.h)
        self.est = self.kf.statePost[:2, 0].copy()
        self.prev = gray.copy()
        return self.est[0], self.est[1], int(self.tracking)


# BEN 2026-11-01
# Change aim: Baseline 1: plain LK median flow, no Kalman, no re-acquisition.
# File: track_video.py
# Function: PlainLk
class PlainLk:
    def __init__(self, gray, cx, cy, w, h):
        self.c, self.w, self.h = np.array([cx, cy], np.float32), w, h
        self.prev, self.pts = gray.copy(), detect_features(gray, self.c, w, h)

    def update(self, gray):
        med, nxt = track_points(self.prev, gray, self.pts)
        st = 0
        if med is not None:
            self.c, self.pts, st = self.c + med, nxt, 1
        else:
            self.pts = np.zeros((0, 1, 2), np.float32)
        if len(self.pts) < 15:
            self.pts = detect_features(gray, self.c, self.w, self.h)
        self.prev = gray.copy()
        return self.c[0], self.c[1], st


# BEN 2026-11-01
# Change aim: Baseline 2: OpenCV MIL tracker.
# File: track_video.py
# Function: Mil
class Mil:
    def __init__(self, frame, cx, cy, w, h):
        self.t = cv2.TrackerMIL_create()
        self.box = (int(round(cx - w / 2)), int(round(cy - h / 2)), w, h)
        self.t.init(frame, self.box)

    def update(self, frame):
        ok, b = self.t.update(frame)
        if ok:
            self.box = b
        return self.box[0] + self.box[2] / 2, self.box[1] + self.box[3] / 2, int(ok)


# BEN 2026-11-01
# Change aim: IoU of two centre-size boxes.
# File: track_video.py
# Function: iou
def iou(a, b):
    x1, x2 = max(a[0] - a[2] / 2, b[0] - b[2] / 2), min(a[0] + a[2] / 2, b[0] + b[2] / 2)
    y1, y2 = max(a[1] - a[3] / 2, b[1] - b[3] / 2), min(a[1] + a[3] / 2, b[1] + b[3] / 2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


# BEN 2026-11-01
# Change aim: Run the three trackers, print the metrics table, write CSV files.
# File: track_video.py
# Function: main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default="run")
    a = ap.parse_args()
    gt = np.genfromtxt(a.gt, delimiter=",", skip_header=1)
    cap = cv2.VideoCapture(a.video)
    names = ["LK + Kalman + template (this repo)", "Plain LK (median flow)", "OpenCV MIL"]
    tracks, ms, trackers, i = [[], [], []], [0.0] * 3, None, 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if i == 0:
            c = (gt[0, 1], gt[0, 2], int(gt[0, 3]), int(gt[0, 4]))
            trackers = [LkKalmanTracker(g, *c), PlainLk(g, *c), Mil(f, *c)]
            for t in tracks:
                t.append((gt[0, 1], gt[0, 2], 1))
        else:
            for k, tr in enumerate(trackers):
                t0 = time.perf_counter()
                tracks[k].append(tr.update(f if k == 2 else g))
                ms[k] += (time.perf_counter() - t0) * 1000
        i += 1
    lines = ["| Tracker | mean error (px) | median (px) | max (px) | success IoU>=0.5 (%) | lost frames (>20 px) | "
             "mean error visible (px) | mean error hidden (px) | ms/frame |", "|---|---|---|---|---|---|---|---|---|"]
    for k, name in enumerate(names):
        tr = np.array(tracks[k])
        n = min(len(tr), len(gt))
        err = np.hypot(tr[1:n, 0] - gt[1:n, 1], tr[1:n, 1] - gt[1:n, 2])
        ious = [iou((tr[j, 0], tr[j, 1], gt[j, 3], gt[j, 4]), (gt[j, 1], gt[j, 2], gt[j, 3], gt[j, 4])) for j in range(1, n)]
        vis = gt[1:n, 5] == 1
        lines.append(f"| {name} | {err.mean():.2f} | {np.median(err):.2f} | {err.max():.2f} | {100 * np.mean(np.array(ious) >= 0.5):.2f} | "
                     f"{int((err > 20).sum())} | {err[vis].mean():.2f} | {(err[~vis].mean() if (~vis).any() else 0):.2f} | {ms[k] / n:.2f} |")
        with open(f"{a.out}/py_track_{['lk_kalman', 'plain_lk', 'mil'][k]}_{a.tag}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["frame", "cx", "cy", "state"])
            for j, r in enumerate(tr):
                w.writerow([j, r[0], r[1], int(r[2])])
    text = "\n".join(lines)
    print(text)
    open(f"{a.out}/py_metrics_{a.tag}.md", "w").write(text + "\n")


if __name__ == "__main__":
    main()
