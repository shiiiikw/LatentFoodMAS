"""Remove records that failed on OpenAI rate-limit/credit errors so a rerun resumes exactly those roots.
Usage: python rclup/scripts/strip_rate_limit_records.py <run_dir> [<run_dir> ...]"""
import json, sys, shutil
from pathlib import Path
for d in sys.argv[1:]:
    p = Path(d) / "records.jsonl"
    rows = [json.loads(l) for l in p.open()]
    keep = [r for r in rows if "RateLimit" not in (r.get("error") or "")]
    shutil.copy(p, p.with_suffix(".jsonl.bak"))
    with p.open("w") as f:
        for r in keep:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (Path(d) / "summary.json").unlink(missing_ok=True)
    print(f"{d}: kept {len(keep)}/{len(rows)}")
