# Crypto-Particle Prediction Challenge
![Project Screenshot](/trial_0002 - frame at 0m1s.jpg)
This repository creates a leakage-resistant forecasting competition with 150
particles distributed across 50 sections. Competitors observe past video and
predict every particle's position one second into an unseen future.

## Download the official dataset

[Download the public challenge dataset from Dropbox](https://www.dropbox.com/scl/fo/ymmcjkuadzsh1ahdmgzr4/ABH3SuHKyIFJPQTT4_EuUX4?rlkey=g8oc9x92ihjvbbyg5yhxny291&st=6pi207cl&e=3&dl=0)

The organizer's private future-position ground truth is intentionally excluded.

## Install

```powershell
python -m pip install -r requirements.txt
```

## Run the simulator

```powershell
python real_water_simulator.py
```

## Verify the dataset pipeline

```powershell
python competition_builder.py --smoke-test --output competition_smoke
```

## Build the official release

```powershell
python competition_builder.py --output competition_release
```

The default release contains 30 minutes of labeled training video and 60
independent five-second test histories. Each test simulation continues
privately for one second after its public video ends.

Publish only:

```text
competition_release/public/
```

Never publish:

```text
competition_release/organizer_private/
```

## Evaluate a submission

Only the organizer runs:

```powershell
python evaluate_submission.py path\to\submission.csv
```

See `COMPETITION_RULES.md` for metrics, submission requirements, and rules.
