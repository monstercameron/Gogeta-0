"""Cloud provider plugins for the cockpit (stdlib only).

A provider knows how to find, rent, list, reach and destroy GPU machines. Everything else (spend approvals, the registry,
setup via cockpit/remote/bootstrap.sh, data, jobs, monitoring, results sync, watchdog guards) is provider-independent and
lives in cockpit/cloud.py and cockpit/fleet.py. The remote side only needs ssh, root and an NVIDIA GPU.

Adding a provider = one new file here that subclasses Provider and calls register(). Normalised shapes:

  offer    {id, gpu_name, num_gpus, dph_total, reliability, cuda_max_good, driver_version, gpu_ram_gb, inet_down_mbps,
            disk_space_gb, geolocation, machine_id, static_ip}
  instance {id, label, gpu_name, num_gpus, dph_total, actual_status, start_date (epoch), ssh_host, ssh_port, machine_id,
            driver_version, image, disk_gb, meta (display-only dict; may hold the IP: never commit it)}

Rules every provider must keep: never read/print/pass API keys (the provider's own CLI/config holds them); label every
instance it creates with the label it is given (the cockpit only manages labels starting `gogeta-cockpit-`); `rent` is
called only by cloud.rent() after the human-approval spend check.
"""
from __future__ import annotations

import abc


class Provider(abc.ABC):
    name = "base"
    spends_money = True          # False for bring-your-own boxes (no approval needed to "rent" them)
    description = ""

    def available(self) -> bool:
        """Is the provider usable from here (CLI present, config readable)?"""
        return True

    @abc.abstractmethod
    def search_offers(self, spec: dict) -> list | dict:
        """spec: {gpu, num_gpus, min_reliability, verified, max_dph, sort, limit, disk_gb, min_cuda}. A list of offers, or {"error": ...}."""

    def offer(self, offer_id, disk_gb: int) -> dict | None:
        """One offer by id (re-read live just before renting)."""
        r = self.search_offers({"id": offer_id, "disk_gb": disk_gb})
        return next((o for o in r if str(o.get("id")) == str(offer_id)), None) if isinstance(r, list) else None

    @abc.abstractmethod
    def rent(self, offer: dict, label: str, disk_gb: int, image: str) -> dict:
        """Create one on-demand instance labelled `label`. {"ok": bool, "id": str, "error": str}. Only cloud.rent() calls this."""

    @abc.abstractmethod
    def instances(self) -> list | None:
        """Every instance on the account (normalised), or None if the listing is unavailable."""

    def instance(self, instance_id) -> dict | None:
        return next((x for x in self.instances() or [] if str(x.get("id")) == str(instance_id)), None)

    def ssh_endpoint(self, instance_id) -> tuple:
        x = self.instance(instance_id) or {}
        return x.get("ssh_host"), x.get("ssh_port")

    def cost(self, instance_id) -> dict:
        import time
        x = self.instance(instance_id) or {}
        dph = float(x.get("dph_total") or 0)
        st = float(x.get("start_date") or time.time())
        return {"dph": dph, "accrued_usd": round(dph * max(0.0, time.time() - st) / 3600, 4)}

    @abc.abstractmethod
    def destroy(self, instance_id) -> dict:
        """Delete the instance. {"ok": bool, "error": str}. Callers check the label first."""

    def balance(self, low_usd: float = 5.0) -> str:
        """'sufficient' | 'low' | 'unknown'. Never an amount or an account identity."""
        return "unknown"

    def describe(self) -> dict:
        return {"name": self.name, "available": self.available(), "spends_money": self.spends_money, "description": self.description}


REGISTRY: dict = {}
_INSTANCES: dict = {}


def register(cls):
    REGISTRY[cls.name] = cls
    return cls


def get(name: str = "vast") -> Provider:
    name = (name or "vast").strip().lower()
    if name not in REGISTRY:
        raise ValueError("unknown provider %r (have: %s)" % (name, ", ".join(sorted(REGISTRY))))
    if name not in _INSTANCES:
        _INSTANCES[name] = REGISTRY[name]()
    return _INSTANCES[name]


def names() -> list:
    return sorted(REGISTRY)


def providers() -> dict:
    """READ-ONLY. The registered cloud providers and whether each is usable here."""
    return {"providers": [get(n).describe() for n in names()],
            "how_to_add": "one file in cockpit/providers/ subclassing Provider + register(); see cockpit/providers/__init__.py"}


from . import ssh, vast  # noqa: E402,F401  (register the built-in plugins)
