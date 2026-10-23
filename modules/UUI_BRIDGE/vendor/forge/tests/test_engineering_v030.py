from __future__ import annotations

import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    init = (ROOT / "equinox_forge" / "__init__.py").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    release = (ROOT / "RELEASE_V0_30.md").read_text(encoding="utf-8")
    contract = json.loads((ROOT / "benchmarks" / "v0_30" / "benchmark_contract.json").read_text(encoding="utf-8"))

    project_version = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, flags=re.M).group(1)
    init_version = re.search(r'__version__\s*=\s*"([^"]+)"', init).group(1)
    # Historical v0.30 freeze artifacts remain immutable, while the live package may advance.
    assert tuple(map(int, project_version.split("."))) >= (0, 30, 0)
    assert init_version == project_version
    assert readme.startswith("# Equinox Forge v0.")
    assert release.startswith("# Equinox Forge v0.30")
    assert '[tool.setuptools.packages.find]' in pyproject
    assert 'include = ["equinox_forge*"]' in pyproject
    assert contract["version"] == "0.30"
    assert contract["frozen"] is True
    assert contract["policy"]["existing_metric_semantics_may_change"] is False
    assert "unknown_rejection_rate" in contract["metric_groups"]["rejection"]
    print("ENGINEERING_V030_OK")


if __name__ == "__main__":
    main()
