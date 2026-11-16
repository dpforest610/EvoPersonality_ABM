"""Re-probe ONLY Gradual and GradualAppeasement against the EXISTING
genotype censuses from the production evolution runs, after fixing their
punishment rule (docs/DEVIATION_gradual_rule.md). Evolution is not
re-run, and the other 5 probe opponents (TFT/AllC/AllD/Random/WSLS) are
left completely untouched.

Reads (Mac-local, unmodified): seed_X_probe_age_G_census.csv.gz, written
by scripts/run_lindgren_markov_sweep.py -- the genotype population at
each probe age, independent of probe noise rate or opponent.

Writes, to a SEPARATE --out-dir (the original production directory is
never modified, for provenance): seed_X_probe_age_G_<noise>_moves.csv.gz /
_summary.csv in the exact same format as run_lindgren_markov_sweep.py's,
but containing ONLY Gradual/GradualAppeasement rows. These are then
spliced into the confirmatory descriptive CSV by
scripts/reanalyze_gradual_rule_fix.py (which also reports the pre-fix
values alongside, per docs/DEVIATION_gradual_rule.md).

Resumable: a (seed, age) whose output files already exist under --out-dir
is skipped, same convention as run_lindgren_markov_sweep.py's checkpointing.

Usage: see docs/MARKOV_STRUCTURE_EVOLUTION_MAC_RUNBOOK.md section 9.
"""

import argparse
import csv
import gzip
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_lindgren_markov_sweep import _fmt_noise, census_path  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evopersonality_abm.lindgren import LindgrenTable  # noqa: E402
from evopersonality_abm.probe import run_probe_duel  # noqa: E402

REPROBE_OPPONENTS = ["Gradual", "GradualAppeasement"]


def reprobe_moves_path(out_dir: Path, seed: int, age: int, noise_rate: float) -> Path:
    return out_dir / f"seed_{seed}_probe_age_{age}_noise{_fmt_noise(noise_rate)}_moves.csv.gz"


def reprobe_summary_path(out_dir: Path, seed: int, age: int, noise_rate: float) -> Path:
    return out_dir / f"seed_{seed}_probe_age_{age}_noise{_fmt_noise(noise_rate)}_summary.csv"


def load_census_genotypes(path: Path) -> List[Tuple[int, int, LindgrenTable]]:
    """[(genotype_id, frequency, LindgrenTable), ...] reconstructed from
    an existing census.csv.gz file -- no evolution re-run needed."""
    out = []
    with gzip.open(path, "rt", newline="") as f:
        for row in csv.DictReader(f):
            table = tuple(int(x) for x in row["table"].split(",") if x != "")
            prehistory = tuple(int(x) for x in row["prehistory"].split(",") if x != "")
            rep = LindgrenTable(int(row["memory_depth"]), table, prehistory)
            out.append((int(row["genotype_id"]), int(row["frequency"]), rep))
    return out


def reprobe_one_seed_age(
    data_dir: Path, out_dir: Path, seed: int, age: int, probe_noise_rate: float,
    probe_repeats: int, probe_rounds: int, main_analysis_rounds: int, master_seed: str,
) -> bool:
    """Returns True if (re)written, False if skipped (missing census or
    already done)."""
    mp = reprobe_moves_path(out_dir, seed, age, probe_noise_rate)
    sp = reprobe_summary_path(out_dir, seed, age, probe_noise_rate)
    if mp.exists() and sp.exists():
        return False

    cpath = census_path(data_dir, seed, age)
    if not cpath.exists():
        return False
    genotypes = load_census_genotypes(cpath)

    rng = random.Random(f"{master_seed}_{seed}_{age}")
    moves_rows: List[Tuple] = []
    summary_rows: List[Tuple] = []
    for gid, freq, rep in genotypes:
        for opponent_name in REPROBE_OPPONENTS:
            own_coop_20 = own_coop_full = opp_coop_20 = opp_coop_full = 0
            for repeat in range(probe_repeats):
                own_moves, opp_moves = run_probe_duel(
                    rep, opponent_name, rng, rounds=probe_rounds, noise_rate=probe_noise_rate
                )
                for r, (om, pm) in enumerate(zip(own_moves, opp_moves), start=1):
                    moves_rows.append((gid, opponent_name, repeat, r, om, pm))
                own_coop_20 += sum(1 for m in own_moves[:main_analysis_rounds] if m == "C")
                opp_coop_20 += sum(1 for m in opp_moves[:main_analysis_rounds] if m == "C")
                own_coop_full += sum(1 for m in own_moves if m == "C")
                opp_coop_full += sum(1 for m in opp_moves if m == "C")
            n = probe_repeats
            summary_rows.append((
                gid, rep.memory_depth, freq, opponent_name, n,
                own_coop_20 / (n * main_analysis_rounds), own_coop_full / (n * probe_rounds),
                opp_coop_20 / (n * main_analysis_rounds), opp_coop_full / (n * probe_rounds),
            ))

    out_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(mp, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["genotype_id", "opponent", "repeat", "round", "own_move", "opp_move"])
        w.writerows(moves_rows)
    with open(sp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "genotype_id", "memory_depth", "frequency", "opponent", "n_repeats",
            "own_coop_rate_20", "own_coop_rate_full", "opp_coop_rate_20", "opp_coop_rate_full",
        ])
        for row in summary_rows:
            w.writerow([row[0], row[1], row[2], row[3], row[4], f"{row[5]:.4f}", f"{row[6]:.4f}", f"{row[7]:.4f}", f"{row[8]:.4f}"])
    return True


def reprobe_one_age(
    data_dir_str: str, out_dir_str: str, seeds: List[int], age: int, probe_noise_rate: float,
    probe_repeats: int, probe_rounds: int, main_analysis_rounds: int, master_seed: str,
) -> Tuple[int, int, int]:
    data_dir, out_dir = Path(data_dir_str), Path(out_dir_str)
    written = skipped_done = skipped_missing = 0
    for seed in seeds:
        mp = reprobe_moves_path(out_dir, seed, age, probe_noise_rate)
        sp = reprobe_summary_path(out_dir, seed, age, probe_noise_rate)
        already_done = mp.exists() and sp.exists()
        had_census = census_path(data_dir, seed, age).exists()
        if reprobe_one_seed_age(
            data_dir, out_dir, seed, age, probe_noise_rate,
            probe_repeats, probe_rounds, main_analysis_rounds, master_seed,
        ):
            written += 1
        elif already_done:
            skipped_done += 1
        elif not had_census:
            skipped_missing += 1
    return age, written, skipped_done, skipped_missing


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=str, required=True, help="original production dir with the existing seed_X_probe_age_G_census.csv.gz files (read-only, never modified)")
    p.add_argument("--out-dir", type=str, required=True, help="separate dir to write the re-probed Gradual/GradualAppeasement moves+summary files to")
    p.add_argument("--probe-noise-rate", type=float, required=True, help="0.01 for the primary role's own-noise probe, 0.0 for the sensitivity role")
    p.add_argument("--seeds", type=int, default=70)
    p.add_argument("--seed-offset", type=int, default=100)
    p.add_argument("--window-start", type=int, default=1000, help="matches the confirmatory analysis window (preregistration section 3) -- the only ages actually used in the report")
    p.add_argument("--window-end", type=int, default=5000)
    p.add_argument("--probe-interval", type=int, default=50)
    p.add_argument("--probe-repeats", type=int, default=30, help="must match run_lindgren_markov_sweep.py's default, used for the original production run")
    p.add_argument("--probe-rounds", type=int, default=100)
    p.add_argument("--main-analysis-rounds", type=int, default=20)
    p.add_argument("--master-seed", type=str, default="gradual_reprobe_v1")
    p.add_argument("--workers", type=int, default=5)
    args = p.parse_args()

    seeds = list(range(args.seed_offset, args.seed_offset + args.seeds))
    ages = list(range(args.window_start, args.window_end + 1, args.probe_interval))

    t0 = time.time()
    total_written = total_skipped_done = total_skipped_missing = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(
                reprobe_one_age, args.data_dir, args.out_dir, seeds, age, args.probe_noise_rate,
                args.probe_repeats, args.probe_rounds, args.main_analysis_rounds, args.master_seed,
            ): age
            for age in ages
        }
        done = 0
        for fut in as_completed(futures):
            age, written, skipped_done, skipped_missing = fut.result()
            total_written += written
            total_skipped_done += skipped_done
            total_skipped_missing += skipped_missing
            done += 1
            print(f"age {age} done ({done}/{len(ages)}): {written} written, {skipped_done} already done, {skipped_missing} missing census", flush=True)

    print(
        f"all ages done in {time.time() - t0:.0f}s -- "
        f"{total_written} (seed,age) units written, {total_skipped_done} already done, "
        f"{total_skipped_missing} missing census (skipped)",
        flush=True,
    )


if __name__ == "__main__":
    main()
