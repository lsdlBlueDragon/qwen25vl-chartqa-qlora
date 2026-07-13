import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


def source_lines(text: str) -> list[str]:
    return [line + "\n" for line in text.rstrip("\n").splitlines()]


def markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_lines(text),
    }


def first_text(cell: dict) -> str:
    return "".join(cell.get("source", [])).strip()


def replace_or_append(cells: list[dict], marker: str, new_cell: dict) -> list[dict]:
    for idx, cell in enumerate(cells):
        if marker in first_text(cell):
            cells[idx] = new_cell
            return cells
    cells.append(new_cell)
    return cells


def main() -> int:
    parser = argparse.ArgumentParser(description="Append or replace a markdown analysis cell in a notebook.")
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--analysis-md", type=Path, required=True)
    parser.add_argument("--marker", required=True)
    parser.add_argument("--backup-dir", type=Path, default=None)
    parser.add_argument("--sync-copy", type=Path, default=None)
    args = parser.parse_args()

    nb = json.loads(args.notebook.read_text(encoding="utf-8"))
    analysis = args.analysis_md.read_text(encoding="utf-8")
    new_cell = markdown_cell(analysis)

    if args.backup_dir:
        args.backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = args.backup_dir / f"{args.notebook.stem}_before_analysis_{stamp}.ipynb"
        shutil.copy2(args.notebook, backup_path)
        print(f"Backup written: {backup_path}")

    original_cells = len(nb["cells"])
    nb["cells"] = replace_or_append(nb["cells"], args.marker, new_cell)
    args.notebook.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Updated notebook: {args.notebook}")
    print(f"Original cells: {original_cells}")
    print(f"Final cells: {len(nb['cells'])}")

    if args.sync_copy:
        args.sync_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.notebook, args.sync_copy)
        print(f"Synced copy: {args.sync_copy}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
