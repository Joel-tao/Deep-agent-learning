"""Verify hashes, extract to a fresh directory and run the packaged demo."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

root = Path(__file__).resolve().parents[1]
archive = root.parent / "Group36_AdaptiveGrasp_Submission.zip"
python = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(sys.executable)
with tempfile.TemporaryDirectory(prefix="grasp_submission_check_") as tmp:
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for info in z.infolist():
            target = (Path(tmp) / info.filename).resolve()
            assert target.is_relative_to(Path(tmp).resolve())
        z.extractall(tmp)
    project = Path(tmp) / "GraspRL"
    checked = 0
    for line in (project / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((project / name).read_bytes()).hexdigest() == expected, name
        checked += 1
    command = [str(python), "demo.py", "--policy", "scripted", "--horizon", "300",
               "--render", "none", "--output", "results/packaged_run.json"]
    result = subprocess.run(command, cwd=project, text=True, capture_output=True, timeout=180)
    (root / "results/package_demo.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    episode = json.loads((project / "results/packaged_run.json").read_text())[0]
    assert episode["success"] and all(episode["stage_ever"])
    payload = {"archive": archive.name, "integrity": "passed", "sha256_files_checked": checked,
               "extracted_demo_success": True, "steps": episode["steps"],
               "python": str(python), "note": "Extracted ZIP executed with fresh reproduced Conda environment."}
    (root / "results/package_validation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
