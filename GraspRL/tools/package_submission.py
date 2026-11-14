"""Create a portable submission ZIP with a SHA-256 manifest; exclude runtimes."""
import hashlib
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parents[1]
excluded_parts = {".git", ".conda", "__pycache__", ".pytest_cache", "tmp"}
excluded_names = {"tests_initial.log", "first_run.log", "first_episode.json", "mujoco_install.log"}
files = sorted(p for p in root.rglob("*") if p.is_file()
               and not excluded_parts.intersection(p.relative_to(root).parts)
               and p.name not in excluded_names and p.suffix != ".pyc"
               and p.name != "SHA256SUMS.txt")
manifest = root / "SHA256SUMS.txt"
manifest.write_text("".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(root).as_posix()}\n" for p in files), encoding="utf-8")
files.append(manifest)
output = root.parent / "Group36_AdaptiveGrasp_Submission.zip"
with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
    for p in files:
        z.write(p, f"GraspRL/{p.relative_to(root).as_posix()}")
with zipfile.ZipFile(output) as z:
    assert z.testzip() is None
    assert "GraspRL/report.pdf" in z.namelist()
print(f"Created and integrity-checked: {output} ({output.stat().st_size / 2**20:.2f} MiB)")
