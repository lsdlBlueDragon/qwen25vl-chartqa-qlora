import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


def source_lines(text: str) -> list[str]:
    if not text:
        return []
    return [line + "\n" for line in text.rstrip("\n").splitlines()]


def markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_lines(text),
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines(text),
    }


def parse_module_markdown(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    cells: list[dict] = []
    buf: list[str] = []
    code_buf: list[str] = []
    in_code = False
    code_lang = ""

    for line in text.splitlines():
        if line.startswith("```"):
            if not in_code:
                if buf and "\n".join(buf).strip():
                    cells.append(markdown_cell("\n".join(buf).strip()))
                buf = []
                in_code = True
                code_lang = line.strip("`").strip()
                code_buf = []
            else:
                if code_lang in {"python", "py"}:
                    cells.append(code_cell("\n".join(code_buf).strip()))
                else:
                    fenced = "```" + code_lang + "\n" + "\n".join(code_buf).strip() + "\n```"
                    cells.append(markdown_cell(fenced))
                in_code = False
                code_lang = ""
                code_buf = []
            continue

        if in_code:
            code_buf.append(line)
        else:
            buf.append(line)

    if in_code:
        raise ValueError(f"Unclosed code fence in {path}")
    if buf and "\n".join(buf).strip():
        cells.append(markdown_cell("\n".join(buf).strip()))

    return cells


def first_text(cell: dict) -> str:
    return "".join(cell.get("source", [])).strip()


def strip_existing_module(cells: list[dict], replace_starts: list[str]) -> list[dict]:
    start = None
    for idx, cell in enumerate(cells):
        text = first_text(cell)
        if any(text.startswith(prefix) for prefix in replace_starts):
            start = idx
            break
    if start is None:
        return cells

    end = len(cells)
    for idx in range(start + 1, len(cells)):
        text = first_text(cells[idx])
        if text.startswith("# Module ") or text.startswith("## Module ") or text.startswith("## 模块 "):
            end = idx
            break
    return cells[:start] + cells[end:]


def main() -> int:
    parser = argparse.ArgumentParser(description="Append or replace a Colab module from markdown in an ipynb.")
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--module-md", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path, default=None)
    parser.add_argument("--sync-copy", type=Path, default=None)
    parser.add_argument(
        "--replace-starts",
        nargs="+",
        default=["# Module 21", "## Module 21", "## 模块 21"],
        help="Cell text prefixes that mark an existing module to replace.",
    )
    args = parser.parse_args()

    if not args.notebook.exists():
        raise FileNotFoundError(args.notebook)
    if not args.module_md.exists():
        raise FileNotFoundError(args.module_md)

    nb = json.loads(args.notebook.read_text(encoding="utf-8"))
    original_cell_count = len(nb["cells"])
    module_cells = parse_module_markdown(args.module_md)
    if not module_cells:
        raise ValueError(f"No cells parsed from {args.module_md}")

    if args.backup_dir:
        args.backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = args.backup_dir / f"{args.notebook.stem}_before_module21_{stamp}.ipynb"
        shutil.copy2(args.notebook, backup_path)
        print(f"Backup written: {backup_path}")

    nb["cells"] = strip_existing_module(nb["cells"], args.replace_starts) + module_cells
    args.notebook.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Updated notebook: {args.notebook}")
    print(f"Original cells: {original_cell_count}")
    print(f"Module cells added: {len(module_cells)}")
    print(f"Final cells: {len(nb['cells'])}")

    if args.sync_copy:
        args.sync_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.notebook, args.sync_copy)
        print(f"Synced copy: {args.sync_copy}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
