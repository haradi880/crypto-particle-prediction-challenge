"""Build the public and private assets for the +1 Second Particle Challenge.

The default build is intentionally large: 30 minutes of labeled training
video and 60 independent test clips. Use --smoke-test to verify the complete
pipeline in about a minute before starting the full build.
"""

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pygame

import real_water_simulator as simulator


SIM_FPS = 60
VIDEO_FPS = 20
FRAME_STRIDE = SIM_FPS // VIDEO_FPS
HORIZON_FRAMES = SIM_FPS
PARTICLE_COUNT = 150


def render_bgr(sim):
    sim.draw()
    rgb = pygame.surfarray.array3d(simulator.screen)
    return cv2.cvtColor(np.transpose(rgb, (1, 0, 2)), cv2.COLOR_RGB2BGR)


def positions(sim):
    return [
        {
            "section_id": p.section_id,
            "color": p.color_name,
            "x": round(float(p.pos[0]), 6),
            "y": round(float(p.pos[1]), 6),
        }
        for p in sim.particles
    ]


def advance(sim, frame_id):
    sim.frame_id = frame_id
    sim.update([0.0, 0.0], 1.0 / SIM_FPS)
    sim.entropy_pool.mix(sim.particles)


def writer(path):
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    result = cv2.VideoWriter(
        str(path), codec, VIDEO_FPS, (simulator.WIDTH, simulator.HEIGHT)
    )
    if not result.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    return result


def build_training(public_dir, duration_seconds):
    print(f"Building {duration_seconds}s labeled training recording...")
    sim = simulator.WaterSimulator()
    video_path = public_dir / "train" / "train.mp4"
    labels_path = public_dir / "train" / "labels.csv"
    video = writer(video_path)

    with labels_path.open("w", newline="", encoding="utf-8") as handle:
        output = csv.writer(handle)
        output.writerow([
            "video_frame", "time_sec", "section_id", "color", "x", "y"
        ])
        total_frames = duration_seconds * SIM_FPS
        video_frame = 0
        for frame_id in range(1, total_frames + 1):
            advance(sim, frame_id)
            if frame_id % FRAME_STRIDE:
                continue
            video.write(render_bgr(sim))
            timestamp = frame_id / SIM_FPS
            for point in positions(sim):
                output.writerow([
                    video_frame, f"{timestamp:.6f}", point["section_id"],
                    point["color"], point["x"], point["y"],
                ])
            video_frame += 1
            if video_frame % (VIDEO_FPS * 60) == 0:
                print(f"  training minutes completed: {video_frame // (VIDEO_FPS * 60)}")
    video.release()
    return video_frame


def build_test(public_dir, private_dir, trial_count, history_seconds):
    print(f"Building {trial_count} independent hidden-future test trials...")
    manifest = []
    truth_rows = []

    for trial_number in range(1, trial_count + 1):
        trial_id = f"trial_{trial_number:04d}"
        sim = simulator.WaterSimulator()
        clip_path = public_dir / "test" / f"{trial_id}.mp4"
        video = writer(clip_path)
        history_frames = history_seconds * SIM_FPS
        video_frame = 0

        for frame_id in range(1, history_frames + 1):
            advance(sim, frame_id)
            if frame_id % FRAME_STRIDE == 0:
                video.write(render_bgr(sim))
                video_frame += 1
        video.release()

        # The public video ends here. Generate the unknown future only after
        # the last observable frame has been committed to disk.
        for offset in range(1, HORIZON_FRAMES + 1):
            advance(sim, history_frames + offset)

        manifest.append({
            "trial_id": trial_id,
            "video": f"test/{trial_id}.mp4",
            "history_seconds": history_seconds,
            "video_fps": VIDEO_FPS,
            "prediction_horizon_seconds": 1.0,
            "expected_rows": PARTICLE_COUNT,
        })
        for point in positions(sim):
            truth_rows.append({"trial_id": trial_id, **point})

        print(f"  completed {trial_id}")

    with (public_dir / "test_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    truth_path = private_dir / "ground_truth.csv"
    with truth_path.open("w", newline="", encoding="utf-8") as handle:
        output = csv.DictWriter(
            handle, fieldnames=["trial_id", "section_id", "color", "x", "y"]
        )
        output.writeheader()
        output.writerows(truth_rows)


def write_templates(public_dir):
    template_path = public_dir / "submission_template.csv"
    manifest = json.loads((public_dir / "test_manifest.json").read_text("utf-8"))
    with template_path.open("w", newline="", encoding="utf-8") as handle:
        output = csv.writer(handle)
        output.writerow(["trial_id", "section_id", "color", "pred_x", "pred_y"])
        for trial in manifest:
            for section_id in range(50):
                for color in ("red", "green", "blue"):
                    output.writerow([trial["trial_id"], section_id, color, "", ""])

    for source_name in (
        "real_water_simulator.py",
        "COMPETITION_RULES.md",
        "requirements.txt",
    ):
        shutil.copy2(source_name, public_dir / source_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="competition_release")
    parser.add_argument("--train-minutes", type=int, default=30)
    parser.add_argument("--test-trials", type=int, default=60)
    parser.add_argument("--history-seconds", type=int, default=5)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        train_seconds, test_trials, history_seconds = 3, 2, 2
    else:
        train_seconds = args.train_minutes * 60
        test_trials, history_seconds = args.test_trials, args.history_seconds

    root = Path(args.output)
    if root.exists():
        if not args.overwrite:
            raise SystemExit(f"{root} exists; use --overwrite to replace it.")
        shutil.rmtree(root)

    public_dir = root / "public"
    private_dir = root / "organizer_private"
    (public_dir / "train").mkdir(parents=True)
    (public_dir / "test").mkdir(parents=True)
    private_dir.mkdir(parents=True)

    training_frames = build_training(public_dir, train_seconds)
    build_test(public_dir, private_dir, test_trials, history_seconds)
    write_templates(public_dir)

    metadata = {
        "name": "+1 Second Crypto-Particle Prediction Challenge",
        "sim_fps": SIM_FPS,
        "video_fps": VIDEO_FPS,
        "width": simulator.WIDTH,
        "height": simulator.HEIGHT,
        "sections": 50,
        "particles_per_section": 3,
        "colors": ["red", "green", "blue"],
        "particle_count": PARTICLE_COUNT,
        "training_duration_seconds": train_seconds,
        "training_video_frames": training_frames,
        "test_trials": test_trials,
        "history_seconds_per_trial": history_seconds,
        "prediction_horizon_seconds": 1.0,
    }
    (public_dir / "dataset.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Complete. Publish only: {public_dir}")
    print(f"Keep secret:          {private_dir}")


if __name__ == "__main__":
    main()
