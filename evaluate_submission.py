"""Organizer-only evaluator for challenge submission CSV files."""

import argparse
import csv
import math
from collections import defaultdict


KEY = ("trial_id", "section_id", "color")


def load_csv(path, prediction):
    result = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["trial_id"], int(row["section_id"]), row["color"])
            if key in result:
                raise SystemExit(f"Duplicate row in {path}: {key}")
            x_name, y_name = ("pred_x", "pred_y") if prediction else ("x", "y")
            try:
                result[key] = (float(row[x_name]), float(row[y_name]))
            except (ValueError, KeyError):
                raise SystemExit(f"Invalid coordinates in {path}: {key}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("submission")
    parser.add_argument(
        "--truth", default="competition_release/organizer_private/ground_truth.csv"
    )
    args = parser.parse_args()

    truth = load_csv(args.truth, False)
    prediction = load_csv(args.submission, True)
    missing, extra = set(truth) - set(prediction), set(prediction) - set(truth)
    if missing or extra:
        raise SystemExit(
            f"Submission keys do not match: {len(missing)} missing, {len(extra)} extra"
        )

    errors = []
    within = defaultdict(int)
    per_trial = defaultdict(list)
    for key, (actual_x, actual_y) in truth.items():
        pred_x, pred_y = prediction[key]
        error = math.hypot(pred_x - actual_x, pred_y - actual_y)
        errors.append(error)
        per_trial[key[0]].append(error)
        for threshold in (1, 5, 10, 20):
            within[threshold] += error <= threshold

    ordered = sorted(errors)
    mean_error = sum(errors) / len(errors)
    median_error = ordered[len(ordered) // 2]
    rmse = math.sqrt(sum(value * value for value in errors) / len(errors))
    score = 100.0 / (1.0 + mean_error)

    print("=== OFFICIAL RESULT ===")
    print(f"Rows evaluated: {len(errors):,}")
    print(f"Mean error:      {mean_error:.6f} px")
    print(f"Median error:    {median_error:.6f} px")
    print(f"RMSE:            {rmse:.6f} px")
    for threshold in (1, 5, 10, 20):
        print(f"Within {threshold:>2}px:      {within[threshold] / len(errors):.4%}")
    print(f"Leaderboard score (higher is better): {score:.6f}")


if __name__ == "__main__":
    main()
