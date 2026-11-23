"""Vast.ai provider plugin. Uses the vastai CLI (Windows exe via WSL interop by default, so the API key stays in the Windows
vastai config); $GOGETA_VASTAI overrides the command (tests: a fake shim). The key is never read, passed or printed here."""
from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path

try:
    from .. import core
except ImportError:  # script mode (cockpit/ on sys.path)
    import core  # type: ignore

from . import Provider, register

WIN_EXE = Path("/mnt/c/Users/mreca/AppData/Roaming/Python/Python312-arm64/Scripts/vastai.exe")


def vast_argv() -> list:
    env = os.environ.get("GOGETA_VASTAI")
    if env:
        return shlex.split(env)
    if WIN_EXE.exists():
        return [str(WIN_EXE)]
    return ["vastai"]


def _child_env() -> dict:
    """Explicit allowlist: the CLI child never inherits the local environment (no tokens, no canaries)."""
    keep = ("PATH", "HOME", "LANG", "LC_ALL", "WSL_INTEROP", "WSL_DISTRO_NAME", "SYSTEMROOT", "FAKE_VAST_STATE", "FAKE_VAST_CALLS")
    e = {k: os.environ[k] for k in keep if k in os.environ}
    e["PYTHONDONTWRITEBYTECODE"] = "1"
    e["WSLENV"] = ""  # pass nothing from Linux into the Windows process
    return e


def run(args: list, timeout: int = 90, raw: bool = True) -> tuple:
    """Run vastai; (rc, parsed JSON or None, redacted text). Refuses any --api-key argument."""
    args = [str(a) for a in args]
    if any(a.startswith(("--api-key", "--api_key")) for a in args):
        raise ValueError("the cockpit never passes an API key: vastai reads its own config")
    argv = vast_argv() + args + (["--raw"] if raw else [])
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=_child_env(),
                           cwd=str(core.DATA if core.DATA.exists() else Path.home()), errors="replace")
        rc, out, err = r.returncode, r.stdout or "", r.stderr or ""
    except subprocess.TimeoutExpired:
        return 124, None, "vastai timed out after %ss" % timeout
    except OSError as ex:
        return 127, None, "vastai not runnable: %s" % ex
    data, s = None, out.strip()
    if raw and s:
        for start in ("[", "{"):
            i = s.find(start)
            if i >= 0:
                try:
                    data = json.loads(s[i:])
                    break
                except ValueError:
                    continue
    text = core.redact((s if data is None else "") + ("\n" + err.strip() if err.strip() else ""))
    return rc, data, core._cap(text, 2000)


def _endpoint(L: dict) -> tuple:
    """Direct SSH (public_ipaddr + ports['22/tcp'][0].HostPort) first; the ssh*.vast.ai proxy refused connections in the pilot."""
    ports = L.get("ports") or {}
    p22 = ports.get("22/tcp") if isinstance(ports, dict) else None
    ip = str(L.get("public_ipaddr") or "").strip()
    try:
        if ip and isinstance(p22, list) and p22 and p22[0].get("HostPort"):
            return ip, int(p22[0]["HostPort"])
        if L.get("ssh_host") and L.get("ssh_port"):
            return str(L["ssh_host"]), int(L["ssh_port"])
    except (TypeError, ValueError):
        pass
    return None, None


def _meta(L: dict) -> dict:
    """Display metadata, ALLOWLISTED (jupyter_token, credit_balance, ... are never copied)."""
    h, p = _endpoint(L)
    f = lambda k, nd=4: core._f(L.get(k), nd)  # noqa: E731
    return {"gpu": L.get("gpu_name"), "num_gpus": L.get("num_gpus"), "vram_gb": core._f((L.get("gpu_totalram") or L.get("gpu_ram") or 0) / 1024, 1),
            "driver": L.get("driver_version"), "cuda_max": L.get("cuda_max_good"), "compute_cap": L.get("compute_cap"), "gpu_mem_bw_gbs": f("gpu_mem_bw", 1),
            "cpu": L.get("cpu_name"), "cpu_cores": L.get("cpu_cores_effective") or L.get("cpu_cores"), "ram_gb": core._f((L.get("cpu_ram") or 0) / 1024, 1),
            "location": L.get("geolocation"), "country": L.get("country_code"), "hosting_type": L.get("hosting_type"), "verification": L.get("verification"),
            "static_ip": L.get("static_ip"), "reliability": f("reliability2"), "inet_up_mbps": f("inet_up", 0), "inet_down_mbps": f("inet_down", 0),
            "dph_total": f("dph_total"), "dph_gpu": f("dph_base"), "dph_disk": f("storage_total_cost"), "bw_up_usd_per_gb": f("inet_up_cost", 5),
            "bw_down_usd_per_gb": f("inet_down_cost", 5), "bw_billed_gb": [core._f(L.get("inet_up_billed"), 3), core._f(L.get("inet_down_billed"), 3)],
            "disk_gb": f("disk_space", 1), "disk_used_gb": f("disk_usage", 1), "image": L.get("image_uuid"),
            "status_msg": core.redact(str(L.get("status_msg") or ""))[:80], "uptime_min": f("uptime_mins", 1), "ip": h, "ssh_port": p}


def _norm_instance(L: dict) -> dict:
    h, p = _endpoint(L)
    return {"id": str(L.get("id")), "label": L.get("label") or "", "gpu_name": L.get("gpu_name"), "num_gpus": L.get("num_gpus") or 1,
            "dph_total": float(L.get("dph_total") or 0), "actual_status": str(L.get("actual_status") or L.get("cur_state") or ""),
            "start_date": L.get("start_date"), "ssh_host": h, "ssh_port": p, "machine_id": str(L.get("machine_id") or ""),
            "driver_version": L.get("driver_version"), "image": L.get("image_uuid"), "disk_gb": L.get("disk_space"), "meta": _meta(L)}


def _norm_offer(o: dict) -> dict:
    return {"id": o.get("id"), "gpu_name": o.get("gpu_name"), "num_gpus": o.get("num_gpus"), "dph_total": core._f(o.get("dph_total"), 4),
            "reliability": core._f(o.get("reliability2", o.get("reliability")), 4), "cuda_max_good": o.get("cuda_max_good"),
            "driver_version": o.get("driver_version"), "gpu_ram_gb": core._f((o.get("gpu_ram") or 0) / 1024, 1),
            "inet_down_mbps": core._f(o.get("inet_down"), 0), "disk_space_gb": core._f(o.get("disk_space"), 0),
            "geolocation": o.get("geolocation"), "machine_id": o.get("machine_id"), "static_ip": o.get("static_ip"), "compute_cap": o.get("compute_cap")}


@register
class VastProvider(Provider):
    name = "vast"
    description = "Vast.ai marketplace (on-demand only until mid-run resume lands)"

    def available(self) -> bool:
        return bool(os.environ.get("GOGETA_VASTAI")) or WIN_EXE.exists()

    def search_offers(self, spec: dict):
        disk = int(spec.get("disk_gb") or 40)
        if spec.get("id") is not None:
            q, extra = "id=%d rentable=true" % int(spec["id"]), ["-n"]
        else:
            gpu = str(spec.get("gpu", "RTX 3090") or "").strip()
            q = ("gpu_name=%s " % gpu.replace(" ", "_") if gpu else "") + "num_gpus=%d reliability>=%s verified=%s rentable=true" % (
                int(spec.get("num_gpus", 1)), float(spec.get("min_reliability", 0.98)), "true" if spec.get("verified", True) else "false")
            if float(spec.get("max_dph") or 0) > 0:
                q += " dph_total<=%s" % float(spec["max_dph"])
            if float(spec.get("min_cuda") or 0) > 0:
                q += " cuda_max_good>=%s" % float(spec["min_cuda"])
            extra = ["-o", spec.get("sort", "dph_total"), "--limit", str(max(1, min(int(spec.get("limit", 20)), 500)))]
        rc, data, text = run(["search", "offers", q] + extra + ["--storage", str(disk)])
        if rc != 0 or not isinstance(data, list):
            return {"error": "vastai search failed (rc %s): %s" % (rc, text[:300]), "query": q}
        return [_norm_offer(o) for o in data]

    def rent(self, offer: dict, label: str, disk_gb: int, image: str) -> dict:
        argv = self.rent_command(offer, label, disk_gb, image)
        rc, data, text = run(argv, timeout=180)
        if isinstance(data, dict) and data.get("success") and data.get("new_contract"):
            return {"ok": True, "id": str(data["new_contract"])}
        return {"ok": False, "error": "create instance failed (rc %s): %s" % (rc, text[:300])}

    @staticmethod
    def rent_command(offer: dict, label: str, disk_gb: int, image: str) -> list:
        return ["create", "instance", str(int(offer["id"])), "--image", image, "--disk", str(int(disk_gb)), "--label", label,
                "--ssh", "--direct", "--cancel-unavail"]

    def instances(self):
        rc, data, _ = run(["show", "instances"], timeout=60)
        if rc != 0 or not isinstance(data, list):
            return None
        return [_norm_instance(x) for x in data]

    def destroy(self, instance_id) -> dict:
        rc, data, text = run(["destroy", "instance", str(instance_id), "-y"], timeout=120)
        if rc != 0 or (isinstance(data, dict) and data.get("success") is False):
            return {"ok": False, "error": "destroy failed (rc %s): %s" % (rc, text[:300])}
        return {"ok": True}

    def balance(self, low_usd: float = 5.0) -> str:
        rc, data, _ = run(["show", "user"], timeout=60)
        cat = "unknown"
        if rc == 0 and isinstance(data, dict) and isinstance(data.get("credit"), (int, float)):
            cat = "sufficient" if float(data["credit"]) >= float(low_usd) else "low"
        del data
        return cat
