"""SQ79 (hardware research): passive GPU telemetry during a campaign: sustained clocks, power, temperature,
throttle reasons, and how fast the GPU cools between campaign items (claims C125, C126).

    python3 experiments/checks/sq_hw_thermal_log.py [--minutes 15] [--period 2]     # CPU only, nvidia-smi only

CPU job: it never touches CUDA (no torch import); it only polls `nvidia-smi --query-gpu`, which is allowed while
a training run holds the GPU slot. Start it right before, or during, a campaign (e.g. K7 / phase 2b) via
`tools/launch_cpu_job.sh --name sq79 -- python3 experiments/checks/sq_hw_thermal_log.py --minutes 15`.
Source: notes/hardware-spark.md, proposal "temperature-aware campaign gap". Patch: notes/patches/campaign-adaptive-gap.patch.

Preliminary observation that motivated it (passive nvidia-smi sampling by the hardware research, 2026-11-22
17:42-17:46, K7 campaign item E10 lr 6e-3 then the 60 s gap; raw CSV in refs/12-hardware/observed/): under load
the SM clock was ~1.8-2.2 GHz at 35-56 W and 65-78 C; after the run ended the GPU was at 59.5 C after ~15 s and
~51 C after ~40 s, then sat idle for the rest of the fixed 60 s gap.

Busy = utilization >= 80%. A "cool-down" starts at a busy -> idle (< 10%) transition.
"""
import argparse
import datetime as dt
import json
import pathlib
import shutil
import statistics
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "checks" / "results"
CHECK_ID, FILE_STEM = "SQ79", "sq_hw_thermal_log"
CLAIMS = ["C125", "C126"]
# ---------------------------------------------------------------------------- written before running
PREDICTION = (
    "Busy samples: median SM clock 1.8-2.2 GHz, median power 38-52 W (the laptop sits near the bottom of its 45-80 W "
    "TDP range in the current Windows power mode), max temperature 76-84 C, power-cap reason present in >= 20% of "
    "busy samples, thermal-slowdown reasons in < 5%. Cool-downs: < 65 C within 30 s of a run ending, so a fixed 60 s "
    "gap leaves the GPU idle and already cool for >= 30 s per campaign item."
)
CRITERION = (
    "C125 (in the current power mode the N1X sustains >= 1.8 GHz median SM clock under our training load while "
    "staying < 85 C, i.e. it is power-limited, not thermally limited) is VERIFIED if the busy median clock >= 1800 "
    "MHz, max temp < 85 C and thermal-slowdown reasons appear in < 5% of busy samples. C126 (an adaptive gap saves "
    ">= 30 s per campaign item) is VERIFIED if the median time to reach 65 C after a run is <= 30 s."
)
NSMI = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
FIELDS = ["timestamp", "temperature.gpu", "utilization.gpu", "clocks.sm", "power.draw", "pstate",
          "clocks_event_reasons.sw_power_cap", "clocks_event_reasons.hw_slowdown",
          "clocks_event_reasons.sw_thermal_slowdown", "clocks_event_reasons.hw_thermal_slowdown"]


def num(s):
    try:
        return float(s.strip().split()[0])
    except Exception:
        return None


def sample():
    out = subprocess.run([NSMI, "--query-gpu=" + ",".join(FIELDS), "--format=csv,noheader,nounits", "-i", "0"],
                         capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
    if not out:
        return None
    v = [x.strip() for x in out[0].split(",")]
    if len(v) != len(FIELDS):
        return None
    r = dict(zip(FIELDS, v))
    return {"t": time.time(), "temp": num(r["temperature.gpu"]), "util": num(r["utilization.gpu"]),
            "sm": num(r["clocks.sm"]), "power": num(r["power.draw"]), "pstate": r["pstate"],
            "power_cap": r["clocks_event_reasons.sw_power_cap"].startswith("Active"),
            "thermal": any(r[k].startswith("Active") for k in ("clocks_event_reasons.sw_thermal_slowdown",
                                                               "clocks_event_reasons.hw_thermal_slowdown",
                                                               "clocks_event_reasons.hw_slowdown"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=15)
    ap.add_argument("--period", type=float, default=2)
    a = ap.parse_args()
    t_start = time.time()
    rows = []
    while time.time() - t_start < a.minutes * 60:
        try:
            s = sample()
            if s:
                rows.append(s)
        except Exception:
            pass
        time.sleep(a.period)
    busy = [r for r in rows if (r["util"] or 0) >= 80]
    cools = []
    i = 1
    while i < len(rows):
        if (rows[i - 1]["util"] or 0) >= 80 and (rows[i]["util"] or 0) < 10:
            t_end, t0 = rows[i - 1]["t"], rows[i - 1]["temp"]
            to65 = next((r["t"] - t_end for r in rows[i:] if r["temp"] is not None and r["temp"] < 65
                         and (r["util"] or 0) < 10), None)
            to60 = next((r["t"] - t_end for r in rows[i:] if r["temp"] is not None and r["temp"] < 60
                         and (r["util"] or 0) < 10), None)
            cools.append({"end_temp": t0, "s_to_65C": to65, "s_to_60C": to60})
        i += 1
    med = lambda xs: statistics.median(xs) if xs else None  # noqa: E731
    data = {"samples": len(rows), "busy_samples": len(busy), "period_s": a.period, "minutes": a.minutes,
            "busy_sm_mhz_median": med([r["sm"] for r in busy if r["sm"]]),
            "busy_power_w_median": med([r["power"] for r in busy if r["power"]]),
            "busy_temp_max": max((r["temp"] for r in busy if r["temp"] is not None), default=None),
            "busy_power_cap_frac": round(sum(r["power_cap"] for r in busy) / len(busy), 3) if busy else None,
            "busy_thermal_frac": round(sum(r["thermal"] for r in busy) / len(busy), 3) if busy else None,
            "pstates_busy": sorted({r["pstate"] for r in busy}), "cooldowns": cools,
            "median_s_to_65C": med([c["s_to_65C"] for c in cools if c["s_to_65C"] is not None])}
    if busy:
        ok125 = (data["busy_sm_mhz_median"] or 0) >= 1800 and (data["busy_temp_max"] or 99) < 85 and \
            (data["busy_thermal_frac"] or 0) < 0.05
        v125 = f"{'VERIFIED' if ok125 else 'REFUTED'}: {data['busy_sm_mhz_median']} MHz, max {data['busy_temp_max']} C"
    else:
        v125 = "inconclusive: no busy samples (was a run active?)"
    m65 = data["median_s_to_65C"]
    v126 = (f"{'VERIFIED' if m65 <= 30 else 'REFUTED'}: median {m65:.0f} s to 65 C over {len(cools)} cool-downs"
            if m65 is not None else "inconclusive: no completed cool-down observed")
    RESULTS.mkdir(parents=True, exist_ok=True)
    rec = {"check": CHECK_ID, "claims": CLAIMS, "prediction": PREDICTION, "criterion": CRITERION,
           "verdicts": {"C125": v125, "C126": v126}, "data": data, "raw": rows,
           "ran_at": dt.datetime.now().isoformat(timespec="seconds"), "runtime_s": round(time.time() - t_start, 1),
           "git_describe": subprocess.run(["git", "-C", str(ROOT), "describe", "--tags", "--always", "--dirty"],
                                          capture_output=True, text=True).stdout.strip(), "device": "nvidia-smi gpu 0"}
    out = RESULTS / f"{FILE_STEM}.json"
    out.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"verdicts": rec["verdicts"], **{k: v for k, v in data.items() if k != "cooldowns"}}, indent=1))
    print(f"-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
