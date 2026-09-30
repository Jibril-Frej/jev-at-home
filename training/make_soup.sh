#!/usr/bin/env bash
# make_soup.sh OUT CALIB ckpt1 ckpt2 ... : average the seeds whose dev accuracy is within 3 points of the best seed,
# fall back to the best seed if the soup is worse (soup_check.py), then fit the shipped temperatures (calibration_u.json).
set -euo pipefail
PY=${PYTHON:-python}
OUT=$1; CALIB=$2; shift 2
SEL=$($PY - "$@" <<'PYEOF'
import json, sys
acc = {c: json.load(open(f"{c}/train_summary.json"))["dev"]["accuracy"] for c in sys.argv[1:]}
best = max(acc.values())
print(" ".join(c for c, a in acc.items() if a >= best - 0.03))
print(json.dumps(acc), file=sys.stderr)
PYEOF
)
echo "$OUT <- $SEL"
$PY soup.py --out "$OUT" $SEL
$PY soup_check.py --soup "$OUT" --seeds $SEL
$PY fit_temperature.py --model "$OUT" --dev "$CALIB" --t-min 0.05
