# Crypto-Particle Prediction Challenge

Can you predict 150 cryptographically driven particles **one second into an
unseen future**?

The simulation contains 50 isolated sections arranged in a `5 x 10` grid.
Every section contains one red, one green, and one blue particle. Their motion
combines inertia, gravity, pressure, viscosity, collisions, boundaries, and
fresh hidden random forces.

You may use computer vision, physics, statistics, machine learning, deep
learning, or any hybrid approach. All models and programming languages are
allowed.

## The challenge

For every test clip:

1. Observe the available history video.
2. Predict the `(x, y)` position of all 150 particles.
3. Your prediction must represent their positions exactly `+1.0` second after
   the final visible frame.
4. Submit one coordinate pair for every section and color.

The evaluated future is not included in the public test videos.

## Download the dataset

[Download the official public dataset from Dropbox](https://www.dropbox.com/scl/fo/ymmcjkuadzsh1ahdmgzr4/ABH3SuHKyIFJPQTT4_EuUX4?rlkey=g8oc9x92ihjvbbyg5yhxny291&st=6pi207cl&e=3&dl=0)

The current release contains:

- 10 minutes of labeled training video
- 12,000 training video frames at 20 FPS
- Coordinates for all 150 particles at every training frame
- 12 independent test-history videos
- 10 seconds of observable history per test
- A one-second hidden prediction horizon
- A ready-to-fill submission template

## Dataset structure

```text
public/
├── train/
│   ├── train.mp4
│   └── labels.csv
├── test/
│   ├── trial_0001.mp4
│   └── ...
├── dataset.json
├── test_manifest.json
├── submission_template.csv
├── real_water_simulator.py
├── requirements.txt
└── COMPETITION_RULES.md
```

Training labels use the full `1400 x 800` video coordinate system:

```csv
video_frame,time_sec,section_id,color,x,y
0,0.050000,0,red,72.125,101.750
```

## Submission format

Fill `submission_template.csv` without changing its rows or column names:

```csv
trial_id,section_id,color,pred_x,pred_y
trial_0001,0,red,72.125,101.750
```

Each test trial requires exactly 150 predictions.

Every entry must include:

- Complete training and inference code
- Filled `submission.csv`
- Model weights, if applicable
- Training data or reproducible generation instructions
- Dependency list
- Raw training and inference outputs
- Hardware and runtime details
- Clear reproduction instructions

## Evaluation

The primary metric is **mean Euclidean position error in pixels**—lower is
better.

Results also report:

- Median pixel error
- Root mean squared error
- Percentage of predictions within 1, 5, 10, and 20 pixels
- Leaderboard score: `100 / (1 + mean pixel error)`

Private ground truth is held by the organizer. Self-reported test scores are
not official.

## Generate additional training data

Competitors may generate unlimited additional data using the supplied
simulator:

```powershell
python -m pip install -r requirements.txt
python real_water_simulator.py
```

You may also use your own video capture, computer-vision pipeline, dataset
format, and model architecture.

## Rules

Reading simulator memory, accessing hidden random-force values, obtaining
private future frames, or leaking test ground truth is prohibited.

See [COMPETITION_RULES.md](COMPETITION_RULES.md) for the complete rules and
deliverable requirements.

Good luck. The future is only one second away—but it is deliberately noisy.
