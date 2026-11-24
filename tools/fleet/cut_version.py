"""Reusable version cutter. Usage: python cut.py spec.json
spec: {"version": "v0.4.16", "title": "...", "entry": "<markdown body after the date line>",
       "files": [...paths to git add...], "body": "commit body", "site": true, "index": true}
Bumps pyproject + README badge/status from the previous version, prepends the DEVLOG entry,
rebuilds site data (stages only docs/data/site-data.js) and the experiment index (tracked-only)."""
import json, pathlib, re, subprocess, sys

R = pathlib.Path(r"C:\Users\mreca\Desktop\Gogeta-0")
CO = "\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"


def git(*a, inp=None):
    p = subprocess.run(["git", "-C", str(R), *a], capture_output=True, text=True, input=inp, encoding="utf-8")
    if p.returncode != 0:
        sys.exit(f"git {' '.join(a)} failed:\n{p.stderr}")
    return p.stdout.strip()


s = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
V, T = s["version"], s["title"]
new = V.lstrip("v")
pp = R / "pyproject.toml"
ptxt = pp.read_text(encoding="utf-8")
old = re.search(r'^version = "([^"]+)"', ptxt, re.M).group(1)
pp.write_text(ptxt.replace(f'version = "{old}"', f'version = "{new}"', 1), encoding="utf-8", newline="\n")
rd = R / "README.md"
rd.write_text(rd.read_text(encoding="utf-8").replace(f"badge/version-v{old}-", f"badge/version-v{new}-")
              .replace(f"> **Status: v{old}.", f"> **Status: v{new}."), encoding="utf-8", newline="\n")
d = R / "DEVLOG.md"
t = d.read_text(encoding="utf-8")
i = t.index("\n## v")
entry = f"## {V}: {T}\n*{s.get('date', '2026-11-22')}*\n\n{s['entry'].strip()}"
d.write_text(t[: i + 1] + entry + "\n\n---\n" + t[i:], encoding="utf-8", newline="\n")
files = ["DEVLOG.md", "pyproject.toml", "README.md", *s["files"]]
if (R / "experiments/claims_list.py").exists():  # AGENTS 5: regenerate the claims list every version
    subprocess.run([sys.executable, "-I", str(R / "experiments/claims_list.py")], check=True, capture_output=True,
                   env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    files += ["experiments/claims-status.md", "experiments/claims_list.py"]
if s.get("site", True):
    subprocess.run([sys.executable, "-I", str(R / "tools/build_site_data.py")], check=True, capture_output=True)
    files += ["docs/data/site-data.js", "docs/experiments", "docs/learn.html", "docs/devlog.html"]
git("add", "--", *files)
if s.get("index", True):
    subprocess.run([sys.executable, "-I", str(R / "tools/build_experiment_index.py"), "--tracked-only"], check=True,
                   capture_output=True)
    git("add", "--", "experiments/INDEX.md")
git("commit", "-q", "-F", "-", inp=f"{V}: {T}\n\n{s['body'].strip()}{CO}")
git("tag", "-a", V, "-m", f"{V}: {T}")
print(git("log", "--oneline", "--decorate", "-1"))
print(git("status", "--short"))
if s.get("push", True):  # AGENTS §3: always push main + version tags to origin
    p = subprocess.run(["git", "-C", str(R), "push", "origin", "main", "--follow-tags"], capture_output=True, text=True)
    print("push rc", p.returncode, (p.stderr or p.stdout).strip()[-400:])
