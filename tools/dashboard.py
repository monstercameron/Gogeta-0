"""Open the local Trackio dashboard for Gogeta-0 (run inside WSL).

    ~/.venvs/gogeta/bin/python tools/dashboard.py [--project gogeta-0] [--port 7861]

Then open http://127.0.0.1:7861 in a Windows browser (WSL forwards localhost).
Local only: public share links/tunnels are disabled.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from gogeta import tracking  # noqa: E402  (sets TRACKIO_DIR before trackio import)

import trackio  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--project", default=tracking.PROJECT)
ap.add_argument("--port", type=int, default=7861)
args = ap.parse_args()

print(f"Trackio data: {tracking.DATA / 'trackio'}")
print(f"Dashboard:    http://127.0.0.1:{args.port}")
trackio.show(project=args.project, host="127.0.0.1", server_port=args.port,
             share=False, open_browser=False, block_thread=True)
