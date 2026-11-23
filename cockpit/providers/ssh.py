"""Bring-your-own SSH box provider ("ssh"): static hosts from a config file, no money spent by the cockpit.

Config: ~/fusiondance-data/cockpit/ssh-hosts.json (data, never committed), written by the human:
  {"hosts": [{"id": "box1", "host": "10.0.0.5", "port": 22, "gpu_name": "RTX 4090", "dph_total": 0.0, "location": "lab"}]}
"Renting" claims a free host (records the label in ~/fusiondance-data/cockpit/ssh-hosts-state.json); "destroying" releases the
claim (the box itself is never touched). Useful for a lab box, a friend's GPU, and for tests.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

try:
    from .. import core
except ImportError:  # script mode
    import core  # type: ignore

from . import Provider, register


def _dir() -> Path:
    p = Path(os.environ.get("GOGETA_COCKPIT_DIR") or core.DATA / "cockpit")
    p.mkdir(parents=True, exist_ok=True)
    return p


def _hosts() -> list:
    try:
        return json.loads((_dir() / "ssh-hosts.json").read_text()).get("hosts", [])
    except (OSError, ValueError):
        return []


def _state() -> dict:
    try:
        return json.loads((_dir() / "ssh-hosts-state.json").read_text())
    except (OSError, ValueError):
        return {}


def _save(st: dict) -> None:
    (_dir() / "ssh-hosts-state.json").write_text(json.dumps(st, indent=1))


@register
class SshProvider(Provider):
    name = "ssh"
    spends_money = False
    description = "bring-your-own boxes from ~/fusiondance-data/cockpit/ssh-hosts.json (no spend; destroy = release the claim)"

    def available(self) -> bool:
        return bool(_hosts())

    def search_offers(self, spec: dict):
        st = _state()
        out = []
        for h in _hosts():
            if spec.get("id") is not None and str(h.get("id")) != str(spec["id"]):
                continue
            if spec.get("gpu") and str(h.get("gpu_name", "")).lower() != str(spec["gpu"]).lower():
                continue
            if str(h.get("id")) in st:
                continue  # claimed
            out.append({"id": h.get("id"), "gpu_name": h.get("gpu_name"), "num_gpus": h.get("num_gpus", 1), "dph_total": float(h.get("dph_total") or 0),
                        "reliability": None, "cuda_max_good": h.get("cuda_max_good"), "driver_version": h.get("driver_version"),
                        "gpu_ram_gb": h.get("gpu_ram_gb"), "inet_down_mbps": None, "disk_space_gb": None, "geolocation": h.get("location"),
                        "machine_id": "ssh:%s" % h.get("id"), "static_ip": True})
        return out

    def rent(self, offer: dict, label: str, disk_gb: int, image: str) -> dict:
        st = _state()
        hid = str(offer["id"])
        if hid in st:
            return {"ok": False, "error": "host %s already claimed by %s" % (hid, st[hid]["label"])}
        st[hid] = {"label": label, "start": time.time()}
        _save(st)
        return {"ok": True, "id": hid}

    @staticmethod
    def rent_command(offer: dict, label: str, disk_gb: int, image: str) -> list:
        return ["claim", str(offer["id"]), "--label", label]

    def instances(self):
        st = _state()
        out = []
        for h in _hosts():
            c = st.get(str(h.get("id")))
            if not c:
                continue
            out.append({"id": str(h["id"]), "label": c["label"], "gpu_name": h.get("gpu_name"), "num_gpus": h.get("num_gpus", 1),
                        "dph_total": float(h.get("dph_total") or 0), "actual_status": "running", "start_date": c["start"], "ssh_host": h.get("host"),
                        "ssh_port": int(h.get("port") or 22), "machine_id": "ssh:%s" % h["id"], "driver_version": h.get("driver_version"),
                        "image": None, "disk_gb": None, "meta": {"gpu": h.get("gpu_name"), "location": h.get("location"), "ip": h.get("host"),
                                                                "ssh_port": int(h.get("port") or 22), "hosting_type": "byo"}})
        return out

    def destroy(self, instance_id) -> dict:
        st = _state()
        st.pop(str(instance_id), None)
        _save(st)
        return {"ok": True, "note": "claim released; the box itself is untouched"}

    def balance(self, low_usd: float = 5.0) -> str:
        return "sufficient"
