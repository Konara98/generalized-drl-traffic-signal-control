"""How repeatable is SUMO with a fixed seed?  (reviewer question 7)

Runs fixed-time control on every test file with the same SUMO seed as evaluate.py,
in several separate processes (the variation appears between processes), and reports
the spread of the average total waiting time per file.

  python sumo_repeatability.py --repeats 5
Output: results/sumo_repeatability.csv and a printed summary.
"""
import argparse, csv, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ONE_PASS = r"""
import common as C, json
rows = []
for s in C.SCENARIOS:
    for i, f in enumerate(C.split_files(s, "test")):
        m = C.run_episode(s, f, "fixed", sumo_seed=C.TEST_SUMO_SEED + i)
        rows.append([s, f.replace("\\", "/").split("/")[-1], m["avg_total_wait"]])
print("JSON" + json.dumps(rows))
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeats", type=int, default=5)
    a = ap.parse_args()
    import json
    import pandas as pd
    rows = []
    for r in range(a.repeats):
        out = subprocess.run([sys.executable, "-c", ONE_PASS], cwd=HERE, capture_output=True, text=True).stdout
        line = next(l for l in out.splitlines() if l.startswith("JSON"))
        rows += [[r] + x for x in json.loads(line[4:])]
        print(f"repeat {r + 1}/{a.repeats} done", flush=True)
    df = pd.DataFrame(rows, columns=["repeat", "scenario", "route_file", "avg_total_wait"])
    df.to_csv(os.path.join(HERE, "results", "sumo_repeatability.csv"), index=False)
    g = df.groupby(["scenario", "route_file"]).avg_total_wait
    per_file = pd.DataFrame({"min": g.min(), "max": g.max(), "range": g.max() - g.min(),
                             "range_pct": 100 * (g.max() - g.min()) / g.mean()})
    print(per_file.round(2).to_string())
    print("\nper scenario: files that varied / max range (s) / max range (%)")
    for s, d in per_file.groupby(level=0):
        print(f"  {s:12} {int((d['range'] > 1e-9).sum())}/{len(d)}  {d['range'].max():.2f}  {d['range_pct'].max():.1f}")


if __name__ == "__main__":
    main()
