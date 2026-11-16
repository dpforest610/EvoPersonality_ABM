#!/bin/bash
# N=1000 scent-model pipeline for an unattended Mac run (Apple Silicon,
# macOS). Written for the bash shipped with macOS (3.2.x) -- no
# associative arrays, no other bash4+ features.
#
# Stages:
#   (a) pytest safety check, then launch M0 production in the
#       background immediately (it needs no calibration), then run the
#       tau-calibration pilot -> p_move pilots -> a short pilot of all
#       5 conditions.
#   (b) mechanically apply docs/PREREGISTRATION_scent_model_N1000.md's
#       criteria (largest tau with mean scent-following index >= 0.5;
#       M1's p_move = M2's observed pilot move rate; M1-low's p_move =
#       M2-low's observed pilot move rate / q) and write the decision
#       + its evidence to docs/SCENT_MODEL_N1000_CALIBRATION_DECISION.md,
#       committed (network-aware: skipped/retried, never fatal).
#   (c) safety gate: if no tau qualifies, or the pytest check failed,
#       stop before production and log why -- M0 (already running,
#       calibration-independent) is left running either way.
#   (d) if the gate passes, launch the remaining 4 production
#       conditions (M1, M2, M1-low, M2-low) in the background too, each
#       30 seeds x 5000 generations, and wait for all 5 to finish.
#
# Usage (recommended -- survives terminal close, prevents sleep):
#   nohup caffeinate -dimsu bash scripts/run_n1000_pipeline_mac.sh \
#       > logs/n1000_pipeline.log 2>&1 &
#
# Progress/stop/resume: see docs printed at the end of this script's
# docstring-equivalent report (also in the chat report accompanying
# this file).

set -u  # unset-variable errors; deliberately NOT -e (one failed step,
        # e.g. a git push, must not kill the whole pipeline)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || { echo "FATAL: cannot cd to repo root $REPO_ROOT"; exit 1; }

mkdir -p logs

PILOT_LOG="logs/n1000_pipeline.log"
log() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*"; }

# ---------------------------------------------------------------- worker count
WORKERS="$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null)"
if [ -z "$WORKERS" ] || [ "$WORKERS" = "0" ]; then
    WORKERS="$(sysctl -n hw.physicalcpu 2>/dev/null)"
fi
if [ -z "$WORKERS" ] || [ "$WORKERS" = "0" ]; then
    WORKERS="$(sysctl -n hw.ncpu 2>/dev/null)"
fi
if [ -z "$WORKERS" ] || [ "$WORKERS" = "0" ]; then
    WORKERS=4
fi
log "Using WORKERS=$WORKERS (performance cores if detected, else physical/logical core fallback)"

# ---------------------------------------------------------------- oversubscription check
# Up to 5 production runs (M0, M1, M2, M1-low, M2-low) run at the same
# time, each with $WORKERS worker processes. If that exceeds the logical
# core count, workers time-share the CPU and every seed runs slower.
N_CONCURRENT_RUNS=5
LOGICAL_CORES="$(sysctl -n hw.logicalcpu 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null)"
TOTAL_WORKERS=$((WORKERS * N_CONCURRENT_RUNS))
if [ -n "$LOGICAL_CORES" ] && [ "$TOTAL_WORKERS" -gt "$LOGICAL_CORES" ]; then
    log "WARNING: $N_CONCURRENT_RUNS concurrent runs x WORKERS=$WORKERS = $TOTAL_WORKERS worker processes," \
        "but this machine has only $LOGICAL_CORES logical cores. Workers will time-share; consider" \
        "WORKERS=$((LOGICAL_CORES / N_CONCURRENT_RUNS > 0 ? LOGICAL_CORES / N_CONCURRENT_RUNS : 1))."
else
    log "Worker budget: $TOTAL_WORKERS worker processes on ${LOGICAL_CORES:-unknown} logical cores."
fi

PY="venv/bin/python3"
if [ ! -x "$PY" ]; then
    PY="python3"
fi
log "Using Python: $PY"

# ---------------------------------------------------------------- network-aware git push helper
git_commit_and_push() {
    # $1 = commit message, remaining args = paths to add.
    # Runs through scripts/git_serial.py, which holds the same cross-process
    # lock as the production runners (so this commit can't collide with
    # theirs on .git/index.lock), commits only these paths, and bounds
    # every git call with a timeout. Never fatal.
    msg="$1"
    shift
    if "$PY" scripts/git_serial.py commit-push -m "$msg" "$@" >>"$PILOT_LOG" 2>&1; then
        log "pushed: $msg"
    else
        log "commit/push did not complete for: $msg (see $PILOT_LOG; run is NOT stopped)"
    fi
    return 0
}

# ---------------------------------------------------------------- (a) safety check
log "=== Safety check: pytest ==="
if ! "$PY" -m pytest -q >>"$PILOT_LOG" 2>&1; then
    log "FATAL: pytest failed. Not starting any run. See $PILOT_LOG for details."
    exit 1
fi
log "pytest OK."

# ---------------------------------------------------------------- M0 production starts immediately
# M0 needs no calibration (no tau, no p_move), so it is safe to start
# now and let it run for the whole pipeline's duration regardless of
# what the calibration gate below decides for the other 4 conditions.
log "=== Starting M0 production in the background (30 seeds x 5000 gens, no calibration needed) ==="
"$PY" scripts/run_scent_spatial_sweep_n1000.py --movement none \
    --workers "$WORKERS" --push-checkpoints \
    >> logs/n1000_production_none.log 2>&1 &
PID_M0=$!
log "M0 production pid=$PID_M0 (log: logs/n1000_production_none.log)"

# ---------------------------------------------------------------- tau/p_move calibration (resumable/idempotent)
# If a previous run of this script already produced a locked-in
# decision, skip straight to reading it -- re-running calibration on
# resume would waste time and risks nothing new (it's deterministic,
# fixed seeds) but is pointless once already decided.
PIPELINE_DECISION_JSON="scripts/output/scent_spatial_sweep_n1000/n1000_pipeline_decision.json"
DECISION_DOC="docs/SCENT_MODEL_N1000_CALIBRATION_DECISION.md"
DECISION_JSON="scripts/output/scent_tau_calibration_n1000/decision.json"

if [ -f "$PIPELINE_DECISION_JSON" ]; then
    log "Found existing $PIPELINE_DECISION_JSON -- resuming with the already-locked-in decision (skipping calibration/pilots)."
    CHOSEN_TAU="$("$PY" -c "import json; print(json.load(open('$PIPELINE_DECISION_JSON'))['chosen_tau'])")"
    P_MOVE="$("$PY" -c "import json; print(json.load(open('$PIPELINE_DECISION_JSON'))['p_move'])")"
    P_MOVE_LOW="$("$PY" -c "import json; print(json.load(open('$PIPELINE_DECISION_JSON'))['p_move_low'])")"
    log "tau=$CHOSEN_TAU p_move=$P_MOVE p_move_low=$P_MOVE_LOW"
else
    log "=== Tau calibration pilot (M2 only, 3 taus x 3 seeds x 300 gens) ==="
    "$PY" scripts/run_scent_tau_calibration_n1000.py --workers "$WORKERS" >> "$PILOT_LOG" 2>&1
    CALIB_EXIT=$?

    if [ $CALIB_EXIT -ne 0 ] || [ ! -f "$DECISION_JSON" ]; then
        log "SAFETY GATE: no candidate tau reached the required scent-following index (or calibration crashed)."
        log "Not proceeding to M1/M2/M1-low/M2-low production. M0 production (pid=$PID_M0) is left running."
        log "See $DECISION_JSON and logs above for the per-tau index values."
        wait "$PID_M0"
        log "M0 production finished. Pipeline stopping here per the safety gate."
        exit 1
    fi

    CHOSEN_TAU="$("$PY" -c "import json; d=json.load(open('$DECISION_JSON')); print(d['selected_tau'])")"
    if [ "$CHOSEN_TAU" = "None" ] || [ -z "$CHOSEN_TAU" ]; then
        log "SAFETY GATE: decision.json has no selected_tau. Not proceeding to M1/M2/M1-low/M2-low production."
        wait "$PID_M0"
        exit 1
    fi
    log "Selected tau = $CHOSEN_TAU"

    # ---- p_move pilots ----
    log "=== M2 pilot at tau=$CHOSEN_TAU (3 seeds x 1000 gens) -- measures p_move for M1, doubles as the M2 pilot report ==="
    "$PY" scripts/run_scent_spatial_sweep_n1000.py --movement scent --tau "$CHOSEN_TAU" \
        --pilot --workers "$WORKERS" >> "$PILOT_LOG" 2>&1

    P_MOVE="$("$PY" - <<PYEOF
import csv, glob, statistics
root = glob.glob("scripts/output/scent_spatial_sweep_n1000/scent_pilot/*")[0]
rates = []
for f in glob.glob(root + "/seed_*_timeseries.csv"):
    rows = list(csv.DictReader(open(f)))
    rates.extend(float(r["move_rate"]) for r in rows)
print(f"{statistics.fmean(rates):.6f}")
PYEOF
)"
    log "M2 pilot observed move_rate (-> p_move for M1) = $P_MOVE"

    log "=== M2-low pilot at tau=$CHOSEN_TAU, q=0.1 (3 seeds x 1000 gens) -- measures p_move for M1-low ==="
    "$PY" scripts/run_scent_spatial_sweep_n1000.py --movement scent --tau "$CHOSEN_TAU" \
        --consider-move-prob 0.1 --out-suffix _low --pilot --workers "$WORKERS" >> "$PILOT_LOG" 2>&1

    M2_LOW_MOVE_RATE="$("$PY" - <<PYEOF
import csv, glob, statistics
root = glob.glob("scripts/output/scent_spatial_sweep_n1000/scent_pilot_low/*")[0]
rates = []
for f in glob.glob(root + "/seed_*_timeseries.csv"):
    rows = list(csv.DictReader(open(f)))
    rates.extend(float(r["move_rate"]) for r in rows)
print(f"{statistics.fmean(rates):.6f}")
PYEOF
)"
    P_MOVE_LOW="$("$PY" -c "print(f'{$M2_LOW_MOVE_RATE / 0.1:.6f}')")"
    log "M2-low pilot observed move_rate = $M2_LOW_MOVE_RATE -> p_move for M1-low (divided by q=0.1) = $P_MOVE_LOW"

    # ---- remaining pilot-report runs (M0, M1, M1-low) ----
    log "=== M0 pilot (3 seeds x 1000 gens, separate from the M0 production already running) ==="
    "$PY" scripts/run_scent_spatial_sweep_n1000.py --movement none --pilot --workers "$WORKERS" >> "$PILOT_LOG" 2>&1

    log "=== M1 pilot at p_move=$P_MOVE (3 seeds x 1000 gens) ==="
    "$PY" scripts/run_scent_spatial_sweep_n1000.py --movement random --p-move "$P_MOVE" \
        --pilot --workers "$WORKERS" >> "$PILOT_LOG" 2>&1

    log "=== M1-low pilot at p_move=$P_MOVE_LOW, q=0.1 (3 seeds x 1000 gens) ==="
    "$PY" scripts/run_scent_spatial_sweep_n1000.py --movement random --p-move "$P_MOVE_LOW" \
        --consider-move-prob 0.1 --out-suffix _low --pilot --workers "$WORKERS" >> "$PILOT_LOG" 2>&1

    log "All 5 conditions' pilots complete."

    # ---- decision record (markdown, for humans) ----
    "$PY" - "$DECISION_JSON" "$CHOSEN_TAU" "$P_MOVE" "$M2_LOW_MOVE_RATE" "$P_MOVE_LOW" "$DECISION_DOC" <<'PYEOF'
import json, sys
decision_json, chosen_tau, p_move, m2_low_rate, p_move_low, out_path = sys.argv[1:]
d = json.load(open(decision_json))
lines = []
lines.append("# N=1000 tau / p_move 決定記録\n")
lines.append("`scripts/run_n1000_pipeline_mac.sh` が機械的に適用した決定。"
              "`docs/PREREGISTRATION_scent_model_N1000.md`の基準どおり。\n")
lines.append("## tau校正(M2のみ、3シード×300世代、gen>=50)\n")
lines.append("| tau | におい追従指数 | 移動率 |")
lines.append("|---|---|---|")
for tau in d["taus_tested"]:
    idx = d["scent_index_by_tau"].get(str(tau), d["scent_index_by_tau"].get(tau))
    mv = d["move_rate_by_tau"].get(str(tau), d["move_rate_by_tau"].get(tau))
    lines.append(f"| {tau} | {idx:.6f} | {mv:.6f} |")
lines.append(f"\n基準を満たした温度(指数>=0.5): {d['qualifying_taus']}")
lines.append(f"\n**選択されたtau = {chosen_tau}**\n")
lines.append("## p_move決定\n")
lines.append(f"M2パイロット(tau={chosen_tau}、3シード×1000世代)の観測move_rate = {p_move}")
lines.append(f"\n**M1のp_move = {p_move}**\n")
lines.append(f"M2-lowパイロット(tau={chosen_tau}、q=0.1、3シード×1000世代)の観測move_rate = {m2_low_rate}")
lines.append(f"\n**M1-lowのp_move = {m2_low_rate} / 0.1 = {p_move_low}**\n")
with open(out_path, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"wrote {out_path}")
PYEOF

    # ---- decision record (JSON, machine-readable, for this script's own resume) ----
    "$PY" -c "
import json
json.dump(
    {'chosen_tau': $CHOSEN_TAU, 'p_move': $P_MOVE, 'p_move_low': $P_MOVE_LOW},
    open('$PIPELINE_DECISION_JSON', 'w'), indent=2,
)
"

    git_commit_and_push "N=1000: tau/p_move calibration decision" \
        "$DECISION_DOC" "$DECISION_JSON" "$PIPELINE_DECISION_JSON" \
        scripts/output/scent_tau_calibration_n1000/ \
        scripts/output/scent_spatial_sweep_n1000/*/*/seed_*_timeseries.csv \
        scripts/output/scent_spatial_sweep_n1000/*/*/seed_*_summary.csv \
        scripts/output/scent_spatial_sweep_n1000/*/*/run_metadata.json

    log "Decision record written to $DECISION_DOC and committed (pushed if network was reachable)."
fi

# ---------------------------------------------------------------- (d) production: remaining 4 conditions
log "=== Safety gate passed. Launching remaining production: M1, M2, M1-low, M2-low (M0 already running) ==="

"$PY" scripts/run_scent_spatial_sweep_n1000.py --movement random --p-move "$P_MOVE" \
    --workers "$WORKERS" --push-checkpoints >> logs/n1000_production_random.log 2>&1 &
PID_M1=$!
log "M1 production pid=$PID_M1"

"$PY" scripts/run_scent_spatial_sweep_n1000.py --movement scent --tau "$CHOSEN_TAU" \
    --workers "$WORKERS" --push-checkpoints >> logs/n1000_production_scent.log 2>&1 &
PID_M2=$!
log "M2 production pid=$PID_M2"

"$PY" scripts/run_scent_spatial_sweep_n1000.py --movement random --p-move "$P_MOVE_LOW" \
    --consider-move-prob 0.1 --out-suffix _low \
    --workers "$WORKERS" --push-checkpoints >> logs/n1000_production_random_low.log 2>&1 &
PID_M1LOW=$!
log "M1-low production pid=$PID_M1LOW"

"$PY" scripts/run_scent_spatial_sweep_n1000.py --movement scent --tau "$CHOSEN_TAU" \
    --consider-move-prob 0.1 --out-suffix _low \
    --workers "$WORKERS" --push-checkpoints >> logs/n1000_production_scent_low.log 2>&1 &
PID_M2LOW=$!
log "M2-low production pid=$PID_M2LOW"

echo "$PID_M0 $PID_M1 $PID_M2 $PID_M1LOW $PID_M2LOW" > logs/n1000_pipeline.pids
log "All 5 production runs launched. PIDs saved to logs/n1000_pipeline.pids"

wait "$PID_M0" "$PID_M1" "$PID_M2" "$PID_M1LOW" "$PID_M2LOW"
log "=== All 5 production conditions finished. Pipeline complete. ==="
