# +1 Second Crypto-Particle Prediction Challenge

## Objective

Predict the `(x, y)` position of all 150 particles exactly one second after
the final frame of each test-history video.

The scene contains 50 sections. Every section contains one red, one green,
and one blue particle.

## Public package

- `train/train.mp4`: synchronized labeled training video.
- `train/labels.csv`: coordinates for every particle and video frame.
- `test/*.mp4`: independent history-only test clips.
- `test_manifest.json`: clip metadata and prediction horizon.
- `submission_template.csv`: required output schema.
- `dataset.json`: dimensions and dataset configuration.

The test videos stop before the evaluated future second. Future frames and
coordinates are never included in the public package.

## Submission format

Submit exactly one row for each `(trial_id, section_id, color)`:

```csv
trial_id,section_id,color,pred_x,pred_y
trial_0001,0,red,72.125,101.750
```

Every trial requires exactly 150 predictions. Coordinates use the full
`1400 x 800` video coordinate system.

## Allowed

- Computer vision, physics, statistics, ML, deep learning, or ensembles.
- Any programming language and hardware.
- Unlimited additional data generated from the published simulator.
- Pretrained general-purpose models, if disclosed.

## Forbidden

- Accessing organizer ground truth.
- Reading simulator process memory or secret random-force values.
- Modifying the official test assets.
- Manually entering positions from leaked future footage.
- Training on any official test future.
- Fabricating training code, outputs, or reported compute.

## Required deliverables

- Complete training and inference source code.
- Dependency/installation file.
- Training data or a reproducible generation script.
- Test data used.
- Model weights.
- Filled `submission.csv`.
- Raw training and inference output.
- Hardware, runtime, and random-seed documentation.
- A README containing one-command reproduction steps.

## Official evaluation

Primary metric: mean Euclidean position error in pixels, lower is better.

The report also includes median error, RMSE, and percentages within 1, 5,
10, and 20 pixels. The display-only leaderboard score is:

```text
100 / (1 + mean_pixel_error)
```

The organizer evaluates submissions against private ground truth. Self-reported
scores are unofficial.

## Important scientific statement

The simulator applies fresh operating-system cryptographic random forces after
the observable history ends. Exact future random values are unavailable to a
competitor. Models can learn physics, inertia, walls, collisions, and position
distributions, but cannot recover future secret entropy from historical data.
