# BEN 2026-11-01
# Change aim: Plot centre error per frame for the three trackers from the CSV files.
# File: plot_results.py
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# BEN 2026-11-01
# Change aim: Load a CSV with a header line into a float array.
# File: plot_results.py
# Function: load
def load(path):
    return np.genfromtxt(path, delimiter=",", skip_header=1)


# BEN 2026-11-01
# Change aim: Draw error-vs-frame for one run and shade the frames where the object is hidden.
# File: plot_results.py
# Function: main
def main(tag, out):
    gt = load(f"results/{tag}.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, label in [("lk_kalman", "LK + Kalman + template"), ("plain_lk", "Plain LK"), ("mil", "OpenCV MIL")]:
        tr = load(f"results/track_{name}_{tag}.csv")
        err = np.hypot(tr[:, 1] - gt[:, 1], tr[:, 2] - gt[:, 2])
        ax.plot(err, label=label)
    hidden = np.where(gt[:, 5] == 0)[0]
    if len(hidden):
        ax.axvspan(hidden[0], hidden[-1], color="grey", alpha=0.25, label="object hidden")
    ax.set_xlabel("frame")
    ax.set_ylabel("centre error (px)")
    ax.set_title(f"Tracking error, {tag}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
