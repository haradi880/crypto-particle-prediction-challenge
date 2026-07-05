# How to Submit

## 1. Prepare your package

Create one ZIP named:

```text
participant-name_particle-challenge_submission.zip
```

It must contain:

```text
submission/
├── submission.csv
├── README.md
├── requirements.txt
├── train.py
├── predict.py
├── model/
├── logs/
│   ├── training_output.txt
│   └── inference_output.txt
└── data_generation/
```

Equivalent project structures are accepted, but the filled
`submission.csv`, complete code, dependencies, outputs, and reproduction
instructions are mandatory.

Do not include the official training videos in your ZIP. Link to any additional
data that cannot reasonably be included.

## 2. Upload the package

Upload the ZIP to GitHub Releases, Dropbox, Google Drive, OneDrive, or another
service that provides a direct organizer-accessible download link.

Keep the link active until the result is published.

## 3. Open the submission form

[Open a Challenge Submission issue](https://github.com/haradi880/crypto-particle-prediction-challenge/issues/new?template=challenge-submission.yml)

Complete every required field and submit the issue. The issue becomes the
official timestamp for your entry.

## 4. Organizer validation

The organizer will:

1. Download and safety-check the package.
2. Validate the CSV schema and all 150 predictions per trial.
3. Run the private evaluator against unseen ground truth.
4. Reproduce the method when required.
5. Post the official metrics in the submission issue.
6. Add the accepted result to the leaderboard.

Malformed, inaccessible, non-reproducible, or rule-breaking entries may be
marked invalid with an explanation.

## Submission CSV

Use the exact official schema:

```csv
trial_id,section_id,color,pred_x,pred_y
trial_0001,0,red,72.125,101.750
```

Do not add an index column, rename fields, omit particles, or add duplicate
rows.
