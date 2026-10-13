"""
Test SpringClassifier on all 62 test images and report accuracy.

Known labels (from visual inspection):
    GOOD: 004,005,015,016,017,018,021,022,023,029,030,
          031,033,040,041,042,043,044,045,046,047,048,
          049,050,051,052,053,054,055,056,057,058,059,
          060,061,062
    BAD:  019,020,024,025,026,027,028,032,034,035,036,
          037,038,039
    UNKNOWN (no label): 001,002,003,006,007,008,009,010,
                        011,012,013,014
"""

import sys
import pathlib
sys.path.insert(0, "D:/spring-defects/src")

from classifier.spring_classifier import SpringClassifier

# ── Ground truth labels ───────────────────────────────────────────────────────
GOOD_NUMS = {4,5,15,16,17,18,21,22,23,29,30,31,33,
             40,41,42,43,44,45,46,47,48,49,50,51,52,
             53,54,55,56,57,58,59,60,61,62}

BAD_NUMS  = {19,20,24,25,26,27,28,32,34,35,36,37,38,39}

TEST_DIR  = pathlib.Path("D:/spring-defects/data/yolo_pose/test/images")

# ── Run classifier ────────────────────────────────────────────────────────────
clf = SpringClassifier()

print(f"\n{'='*65}")
print(f"  {'Image':<22} {'GT':<6} {'Predicted':<10} {'Conf':>6}  {'OK?'}")
print(f"{'='*65}")

tp = fp = tn = fn = 0
unknown_count = 0

for img_path in sorted(TEST_DIR.iterdir()):
    stem   = img_path.stem                    # e.g. "spring_004"
    number = int(stem.split("_")[1])
    result = clf.predict(str(img_path))

    verdict = result["verdict"]
    conf    = result["confidence"]
    conf_str = f"{conf:.2f}" if conf is not None else "N/A"

    if number in GOOD_NUMS:
        gt = "GOOD"
        if verdict == "GOOD":
            ok = "✓"; tn += 1
        elif verdict == "MANUAL_CHECK":
            ok = "~"; fp += 1   # goes to human check
        else:
            ok = "✗"; fp += 1   # wrongly auto-rejected
    elif number in BAD_NUMS:
        gt = "BAD"
        if verdict == "BAD":
            ok = "✓"; tp += 1
        elif verdict == "MANUAL_CHECK":
            ok = "~"; tp += 1   # caught for inspection = acceptable
        else:
            ok = "✗"; fn += 1   # missed completely = dangerous
    else:
        gt = "???"
        ok = "-"
        unknown_count += 1

    print(f"  {img_path.name:<22} {gt:<6} {verdict:<10} {conf_str:>6}  {ok}")

# ── Summary ───────────────────────────────────────────────────────────────────
total_labelled = tp + fp + tn + fn
accuracy  = (tp + tn) / total_labelled * 100 if total_labelled else 0
precision = tp / (tp + fp) * 100 if (tp + fp) else 0
recall    = tp / (tp + fn) * 100 if (tp + fn) else 0

print(f"\n{'='*65}")
print(f"  LABELLED IMAGES: {total_labelled}  "
      f"(GOOD={tn+fp}, BAD={tp+fn}, UNKNOWN={unknown_count})")
print(f"{'='*65}")
print(f"  TP (BAD  correctly detected) : {tp}")
print(f"  FP (GOOD wrongly flagged)    : {fp}")
print(f"  TN (GOOD correctly passed)   : {tn}")
print(f"  FN (BAD  missed)             : {fn}")
print(f"  Accuracy  : {accuracy:.1f}%")
print(f"  Precision : {precision:.1f}%")
print(f"  Recall    : {recall:.1f}%")
print(f"{'='*65}")