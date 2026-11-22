"""Record an experiment's outcome in experiments/experiments.json.

    python tools/record_result.py E0 --status done --version v0.3.0 \
        --config configs/e0.yaml --summary ~/fusiondance-data/runs/<run>/summary.json \
        [--set val_loss=2.31 --set note="3 seeds, mean"]

--summary merges the run's final metrics (summary.json "metrics") into `results`;
--set adds or overrides single results. Afterwards, run tools/build_site_data.py.
"""
import argparse
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
REG = ROOT / "experiments" / "experiments.json"
STATUSES = {"planned", "running", "done", "failed", "abandoned"}


def parse_value(v: str):
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", help="experiment id, e.g. E0")
    ap.add_argument("--status", choices=sorted(STATUSES))
    ap.add_argument("--version")
    ap.add_argument("--config")
    ap.add_argument("--summary", help="path to a run's summary.json")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--notes")
    args = ap.parse_args()

    reg = json.loads(REG.read_text(encoding="utf-8"))
    exp = next((e for e in reg if e["id"] == args.experiment), None)
    if exp is None:
        raise SystemExit(f"No experiment {args.experiment!r} in {REG.name}")

    if args.status:
        exp["status"] = args.status
    if args.version:
        exp["version"] = args.version
    if args.config:
        exp["config"] = args.config
    if args.notes:
        exp["notes"] = args.notes
    if args.summary:
        summary = json.loads(pathlib.Path(args.summary).expanduser().read_text(encoding="utf-8"))
        exp.setdefault("results", {}).update(summary.get("metrics", {}))
    for kv in args.set:
        k, _, v = kv.partition("=")
        exp.setdefault("results", {})[k.strip()] = parse_value(v.strip())

    REG.write_text(json.dumps(reg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"{exp['id']} -> status={exp['status']} version={exp['version']} results={exp.get('results')}")
    print("Next: python tools/build_site_data.py")


if __name__ == "__main__":
    main()
