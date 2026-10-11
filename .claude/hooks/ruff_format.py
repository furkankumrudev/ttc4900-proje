"""PostToolUse hook: Edit/Write sonrası SADECE değişen .py dosyasını `ruff format` ile formatlar.

Claude Code, araç çağrısının bilgisini stdin'den JSON olarak verir; dosya yolu
`tool_input.file_path` alanındadır. Ruff bulunamazsa ya da herhangi bir hata olursa betik
sessizce çıkar; hook hiçbir zaman Claude'un işini engellemez.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PYTHON_SUFFIXES = {".py", ".pyi"}


def find_ruff(project_dir: Path) -> str | None:
    """Önce projenin .venv'indeki ruff'ı, yoksa PATH'tekini döndürür."""
    for candidate in (
        project_dir / ".venv" / "Scripts" / "ruff.exe",  # Windows
        project_dir / ".venv" / "bin" / "ruff",  # Linux/macOS
    ):
        if candidate.is_file():
            return str(candidate)
    return shutil.which("ruff")


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except ValueError:
        return

    file_path = (payload.get("tool_input") or {}).get("file_path")
    if not file_path:
        return
    path = Path(file_path)
    if path.suffix not in PYTHON_SUFFIXES or not path.is_file():
        return

    project_dir = Path(os.environ.get("CLAUDE_PROJECT_DIR") or payload.get("cwd") or ".")
    ruff = find_ruff(project_dir)
    if ruff is None:
        return

    try:
        subprocess.run(
            [ruff, "format", "--quiet", str(path)],
            cwd=project_dir,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return


if __name__ == "__main__":
    main()
    sys.exit(0)
