"""Train and evaluate a one-second-ahead particle position predictor.

Run the simulator long enough to collect logs, close it, then run:
    python train_particle_predictor.py

The split is chronological, so future frames never leak into training.
"""

import glob
import json
import math
import os

import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier
except ImportError as exc:
    raise SystemExit("Install the model dependency: pip install scikit-learn") from exc


LOG_ROOT = "simulation_logs"
FORECAST_SECONDS = 1.0
GRID_SIZE = 10
TEST_FRACTION = 0.25


def load_frames():
    sessions = [
        path for path in glob.glob(os.path.join(LOG_ROOT, "*"))
        if os.path.isdir(path) and glob.glob(os.path.join(path, "frame_*.json"))
    ]
    if not sessions:
        raise SystemExit(
            "No new-format session found. Run real_water_simulator.py first."
        )
    session = max(sessions, key=os.path.getmtime)
    frames = []
    for path in sorted(glob.glob(os.path.join(session, "frame_*.json"))):
        with open(path, encoding="utf-8") as handle:
            frames.append(json.load(handle))
    if len(frames) < 100:
        raise SystemExit("Collect at least 100 logged frames before training.")
    return frames, session


def make_rows(frames):
    interval = float(np.median(np.diff([f["time_sec"] for f in frames])))
    horizon = max(1, round(FORECAST_SECONDS / interval))
    features, labels, metadata = [], [], []

    for i in range(1, len(frames) - horizon):
        previous, current, future = frames[i - 1], frames[i], frames[i + horizon]
        for section_id, colors in current["sections"].items():
            for color_index, color in enumerate(("red", "green", "blue")):
                p0 = previous["sections"][section_id][color]
                p1 = colors[color]
                p2 = future["sections"][section_id][color]
                # Section-local normalized coordinates make all 50 boxes
                # usable by one model. Section geometry is fixed by simulator.
                section = int(section_id)
                row, col = divmod(section, 10)
                left, top = 20 + col * (128.8 + 8), 20 + row * (145.6 + 8)
                width, height = 128.8, 145.6
                nx, ny = (p1["x"] - left) / width, (p1["y"] - top) / height
                px, py = (p0["x"] - left) / width, (p0["y"] - top) / height
                tx = min(GRID_SIZE - 1, max(0, int((p2["x"] - left) / width * GRID_SIZE)))
                ty = min(GRID_SIZE - 1, max(0, int((p2["y"] - top) / height * GRID_SIZE)))
                features.append([
                    section / 49, color_index / 2, nx, ny, px, py,
                    p1["vx"] / 500, p1["vy"] / 500,
                    p1["radius_nm"] / 70, p1["density"],
                ])
                labels.append(ty * GRID_SIZE + tx)
                metadata.append((section, color, p2["x"], p2["y"]))
    return np.asarray(features), np.asarray(labels), metadata, horizon, interval


def main():
    frames, session = load_frames()
    x, y, metadata, horizon, interval = make_rows(frames)
    # Every timestamp contributes 150 adjacent rows; split only at a complete
    # timestamp boundary to prevent the same moment entering both sets.
    rows_per_time = 150
    time_count = len(x) // rows_per_time
    train_times = max(1, math.floor(time_count * (1 - TEST_FRACTION)))
    split = train_times * rows_per_time

    model = RandomForestClassifier(
        n_estimators=250, min_samples_leaf=3, n_jobs=-1,
        class_weight="balanced_subsample", random_state=42,
    )
    model.fit(x[:split], y[:split])
    predicted = model.predict(x[split:])
    accuracy = float(np.mean(predicted == y[split:]))

    print(f"Session: {session}")
    print(f"Frames loaded: {len(frames)}")
    print(f"Forecast: {horizon} logs = {horizon * interval:.3f} seconds")
    print(f"Training examples: {split:,}")
    print(f"Test examples: {len(y) - split:,}")
    print(f"Exact 10x10 destination-cell accuracy: {accuracy:.2%}")
    print(f"Uniform random baseline: {1/(GRID_SIZE**2):.2%}")
    print("Perfect accuracy is impossible because future forces are generated")
    print("from secret OS entropy after the prediction is made.")


if __name__ == "__main__":
    main()
