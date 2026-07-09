"""Compare global-tracking logs recorded by server_low_level_g1_sim.py --log_recovery.

Typical A/B protocol (same motion file per condition, one motion per sim run;
the scripted push makes the drift the recovery has to fix):

    # terminal A, once per run:  python server_high_level_motion_lib.py ...
    # terminal B:
    python server_low_level_g1_sim.py --log_recovery logs/baseline.npz --push_force 80
    python server_low_level_g1_sim.py --log_recovery logs/pos.npz     --push_force 80 --kp_recovery 1.0
    python server_low_level_g1_sim.py --log_recovery logs/posyaw.npz  --push_force 80 --kp_recovery 1.0 --kp_yaw_recovery 1.0

    python data_utils/compare_recovery.py logs/baseline.npz logs/pos.npz logs/posyaw.npz \
        --labels baseline pos-only pos+yaw --out logs/compare

Outputs a metrics table (stdout + metrics.csv) and overlay plots
(compare_recovery.png) in the --out directory.
"""
import argparse
import csv
import os

import numpy as np

# fixed categorical order (colorblind-safe as ordered; do not cycle/reorder)
SERIES_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
                 "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
INK = "#33322e"
INK_MUTED = "#5f5e56"
GRID = "#e8e8e6"
REFERENCE = "#9a9a94"


def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def load_log(path):
    with np.load(path) as f:
        data = f["data"]
        cols = {name: data[:, k] for k, name in enumerate(f["columns"])}
        meta_keys = ("kp_recovery", "kp_yaw_recovery", "kd_recovery", "kd_yaw_recovery",
                     "push_force", "push_time", "push_duration")
        meta = {k: float(f[k]) if k in f.files else 0.0 for k in meta_keys}
    if data.shape[0] == 0:
        raise ValueError(f"{path}: empty log (did the motion server publish ref_root_pose_g1?)")
    run = {
        "t": cols["t"],
        "target_xy": np.stack([cols["target_x"], cols["target_y"]], axis=1),
        "robot_xy": np.stack([cols["robot_x"], cols["robot_y"]], axis=1),
        "meta": meta,
    }
    run["pos_err"] = np.linalg.norm(run["target_xy"] - run["robot_xy"], axis=1)
    run["yaw_err"] = wrap_to_pi(cols["target_yaw"] - cols["robot_yaw"])
    return run


def time_to_recover(t, pos_err, push_end, threshold, sustain=1.0):
    """Seconds from push end until pos_err stays below threshold for `sustain` s."""
    below = pos_err < threshold
    for k in np.where(t >= push_end)[0]:
        window = (t >= t[k]) & (t <= t[k] + sustain)
        if below[k] and below[window].all():
            return t[k] - push_end
    return np.nan


def compute_metrics(run, threshold, sustain):
    t, pos_err, yaw_err = run["t"], run["pos_err"], np.abs(run["yaw_err"])
    m = {
        "pos_err_mean [m]": pos_err.mean(),
        "pos_err_max [m]": pos_err.max(),
        "pos_err_final [m]": pos_err[t >= t[-1] - 1.0].mean(),
        "yaw_err_mean [deg]": np.rad2deg(yaw_err.mean()),
        "yaw_err_max [deg]": np.rad2deg(yaw_err.max()),
    }
    meta = run["meta"]
    if meta["push_force"] != 0.0:
        push_end = meta["push_time"] + meta["push_duration"]
        m["peak_after_push [m]"] = pos_err[t >= meta["push_time"]].max()
        m["recovery_time [s]"] = time_to_recover(t, pos_err, push_end, threshold, sustain)
    return m


def style_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.xaxis.label.set_color(INK_MUTED)
    ax.yaxis.label.set_color(INK_MUTED)
    ax.title.set_color(INK)


def shade_push(ax, meta):
    if meta["push_force"] != 0.0:
        ax.axvspan(meta["push_time"], meta["push_time"] + meta["push_duration"],
                   color="#c3c2b7", alpha=0.3, linewidth=0, label="push")


def plot_runs(runs, labels, threshold, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.4, 1], hspace=0.35, wspace=0.25)
    ax_pos = fig.add_subplot(gs[0, 0])
    ax_yaw = fig.add_subplot(gs[1, 0], sharex=ax_pos)
    ax_traj = fig.add_subplot(gs[:, 1])

    shade_push(ax_pos, runs[0]["meta"])
    shade_push(ax_yaw, runs[0]["meta"])
    ax_pos.axhline(threshold, color=REFERENCE, linewidth=1.2, linestyle="--")
    ax_pos.annotate(f"recovered < {threshold:g} m", xy=(0.99, threshold),
                    xycoords=("axes fraction", "data"), ha="right", va="bottom",
                    fontsize=8, color=INK_MUTED)

    for run, label, color in zip(runs, labels, SERIES_COLORS):
        ax_pos.plot(run["t"], run["pos_err"], color=color, linewidth=1.8, label=label)
        ax_yaw.plot(run["t"], np.rad2deg(run["yaw_err"]), color=color, linewidth=1.8)
        ax_traj.plot(run["robot_xy"][:, 0], run["robot_xy"][:, 1],
                     color=color, linewidth=1.8)

    # all runs share the sim world frame and spawn point, so one reference path suffices
    ref = runs[0]["target_xy"]
    ax_traj.plot(ref[:, 0], ref[:, 1], color=REFERENCE, linewidth=1.4,
                 linestyle="--", label="reference")
    ax_traj.plot(ref[0, 0], ref[0, 1], "o", color=REFERENCE, markersize=5)

    ax_pos.set_title("Global position tracking error", fontsize=11, loc="left")
    ax_pos.set_ylabel("error [m]")
    ax_yaw.set_title("Heading error", fontsize=11, loc="left")
    ax_yaw.set_ylabel("error [deg]")
    ax_yaw.set_xlabel("time since motion start [s]")
    ax_traj.set_title("Root trajectory (world frame)", fontsize=11, loc="left")
    ax_traj.set_xlabel("x [m]")
    ax_traj.set_ylabel("y [m]")
    ax_traj.set_aspect("equal", adjustable="datalim")

    for ax in (ax_pos, ax_yaw, ax_traj):
        style_axes(ax)
    handles = (ax_pos.get_legend_handles_labels()[0]
               + ax_traj.get_legend_handles_labels()[0])
    names = (ax_pos.get_legend_handles_labels()[1]
             + ax_traj.get_legend_handles_labels()[1])
    # below the figure so it can never collide with curves or titles
    fig.legend(handles, names, frameon=False, fontsize=9, labelcolor=INK,
               loc="upper center", bbox_to_anchor=(0.5, 0.02),
               ncol=min(len(names), 6), handlelength=1.6, columnspacing=1.2)

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", nargs="+", help=".npz logs from --log_recovery")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="one label per log (default: filename stems)")
    parser.add_argument("--out", default="recovery_compare",
                        help="output directory for plots and metrics.csv")
    parser.add_argument("--recover_threshold", type=float, default=0.1,
                        help="position error [m] under which the robot counts as recovered")
    parser.add_argument("--recover_sustain", type=float, default=1.0,
                        help="how long [s] the error must stay under the threshold")
    args = parser.parse_args()

    labels = args.labels or [os.path.splitext(os.path.basename(p))[0] for p in args.logs]
    if len(labels) != len(args.logs):
        parser.error(f"got {len(args.logs)} logs but {len(labels)} labels")
    if len(args.logs) > len(SERIES_COLORS):
        parser.error(f"at most {len(SERIES_COLORS)} runs per comparison")

    runs = [load_log(p) for p in args.logs]
    metrics = [compute_metrics(r, args.recover_threshold, args.recover_sustain) for r in runs]

    os.makedirs(args.out, exist_ok=True)

    # metrics table: runs as columns, metrics as rows
    metric_names = max(metrics, key=len).keys()
    name_w = max(len(n) for n in metric_names)
    col_w = max(12, *(len(l) for l in labels)) + 2
    fmt = lambda v: "--" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"
    print("\n" + " " * name_w + "".join(l.rjust(col_w) for l in labels))
    for name in metric_names:
        row = [fmt(m.get(name)) for m in metrics]
        print(name.ljust(name_w) + "".join(v.rjust(col_w) for v in row))
    for run, label in zip(runs, labels):
        meta = run["meta"]
        print(f"{label}: kp={meta['kp_recovery']:g}, kp_yaw={meta['kp_yaw_recovery']:g}, "
              f"kd={meta['kd_recovery']:g}, kd_yaw={meta['kd_yaw_recovery']:g}, "
              f"push={meta['push_force']:g} N @ {meta['push_time']:g}s")

    csv_path = os.path.join(args.out, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "kp_recovery", "kp_yaw_recovery", "push_force", *metric_names])
        for run, label, m in zip(runs, labels, metrics):
            meta = run["meta"]
            writer.writerow([label, meta["kp_recovery"], meta["kp_yaw_recovery"],
                             meta["push_force"], *(m.get(n, "") for n in metric_names)])

    plot_path = os.path.join(args.out, "compare_recovery.png")
    plot_runs(runs, labels, args.recover_threshold, plot_path)
    print(f"\nSaved {plot_path} and {csv_path}")


if __name__ == "__main__":
    main()
