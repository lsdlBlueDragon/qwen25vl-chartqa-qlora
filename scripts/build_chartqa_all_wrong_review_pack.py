"""Build a self-contained review pack for ChartQA all-runs-wrong samples.

The main output is an HTML file intended for upload to a web chat for
multimodal review. When source images are available, they are embedded as
base64 JPEGs so the file does not depend on local paths.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
from collections import Counter, defaultdict
from datetime import date
from io import BytesIO
from pathlib import Path


RUN_ORDER = [
    "baseline_default",
    "standard_steps100",
    "standard_numeric_final",
    "experiment_a_steps200",
    "experiment_b_calcnum",
    "experiment_d_hardmix",
    "experiment_f_steps250_r16a32",
]


DEFAULT_IMAGE_ROOT_CANDIDATES = [
    Path(r"G:\我的云端硬盘\qwen25vl-chartqa-qlora\data\processed\chartqa_val_full_sft_1920_images"),
    Path(r"G:\My Drive\qwen25vl-chartqa-qlora\data\processed\chartqa_val_full_sft_1920_images"),
    Path("data/processed/chartqa_val_full_sft_1920_images"),
]


PROJECT_CONTEXT = """
本文件用于复核 Qwen2.5-VL ChartQA QLoRA 项目的 full-val all-runs-wrong 样本。

项目目标：构建一个可复现的多模态图表问答微调与评估工作流，用于工程作品集和面试展示。当前主线模型是 Qwen/Qwen2.5-VL-3B-Instruct，数据集是 HuggingFaceM4/ChartQA，方法是 baseline 推理、ChartQA SFT 数据转换、3B QLoRA adapter 微调、full validation 对比、错误分析和后续诊断。

当前进展：3B full validation 已完成 baseline、standard adapter、numeric-final control、steps200、calcnum、hardmix、steps250/r16/a32 共七个 run。最佳 relaxed run 是 experiment_d_hardmix，77.86%，baseline_default 是 75.94%；最佳 exact run 是 experiment_f_steps250_r16a32，69.48%。全 run oracle relaxed 为 1595/1920 = 83.07%，仍有 325 个样本所有 run 都错。

本文件关注：这 325 个 all-runs-wrong 样本。当前初始审计认为剩余瓶颈主要不是输出格式或单个 adapter 偶然失败，而是图表 grounding、坐标/日期/图例读取、数值尺度、计数、排序和多步计算。

复核目标：请检查每个样本的图像、问题、reference、七个 run 的预测、primary_error_type 和 audit_tags，判断当前标签是否合理；特别标出数据/标注/evaluator 问题、可通过更高分辨率改善的问题、需要 OCR/table extraction/derendering 的问题，以及纯计算或推理错误。

下一步规划：用人工/模型复核后的标签，构建一个小型 stratified hard subset；优先做 resolution ablation 和 chart-grounding/OCR/derendering 诊断，而不是继续盲目增加 LoRA steps/rank。
""".strip()


def load_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_image_root(explicit: str | None) -> Path | None:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend(DEFAULT_IMAGE_ROOT_CANDIDATES)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def image_to_data_url(path: Path, max_side: int, jpeg_quality: int) -> tuple[str | None, dict]:
    if not path.exists():
        return None, {"status": "missing", "path": str(path)}

    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - environment dependent
        raw = path.read_bytes()
        suffix = path.suffix.lower()
        mime = "image/png" if suffix == ".png" else "image/jpeg"
        return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}", {
            "status": "embedded_original",
            "path": str(path),
            "bytes": len(raw),
            "warning": f"Pillow unavailable: {exc}",
        }

    with Image.open(path) as image:
        image = image.convert("RGB")
        original_size = image.size
        if max(image.size) > max_side:
            image.thumbnail((max_side, max_side))
        rendered_size = image.size
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
        data = buffer.getvalue()

    return f"data:image/jpeg;base64,{base64.b64encode(data).decode('ascii')}", {
        "status": "embedded_jpeg",
        "path": str(path),
        "original_size": original_size,
        "rendered_size": rendered_size,
        "bytes": len(data),
    }


def esc(value: object) -> str:
    return html.escape(str(value if value is not None else ""))


def make_summary(records: list[dict]) -> dict:
    return {
        "total": len(records),
        "primary_error_type_counts": dict(Counter(row["primary_error_type"] for row in records)),
        "tag_counts": dict(Counter(tag for row in records for tag in row.get("audit_tags", []))),
        "human_or_machine_counts": dict(Counter(str(row.get("human_or_machine")) for row in records)),
        "high_consensus_wrong_count": sum(
            "shared_wrong_consensus" in row.get("audit_tags", []) for row in records
        ),
    }


def render_summary_tables(summary: dict) -> str:
    def table_from_counts(title: str, counts: dict) -> str:
        rows = []
        total = sum(counts.values()) or 1
        for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            rows.append(
                f"<tr><td>{esc(key)}</td><td>{count}</td><td>{count / total:.2%}</td></tr>"
            )
        return (
            f"<h3>{esc(title)}</h3>"
            "<table class='summary'><thead><tr><th>label</th><th>count</th><th>share</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )

    return "\n".join(
        [
            f"<p><strong>Total all-runs-wrong samples:</strong> {summary['total']}</p>",
            f"<p><strong>High consensus wrong:</strong> {summary['high_consensus_wrong_count']}</p>",
            table_from_counts("Primary Error Type Counts", summary["primary_error_type_counts"]),
            table_from_counts("Human/Machine Counts", summary["human_or_machine_counts"]),
            table_from_counts("Secondary Tag Counts", summary["tag_counts"]),
        ]
    )


def render_prediction_table(predictions: dict) -> str:
    rows = []
    for run in RUN_ORDER:
        rows.append(f"<tr><td>{esc(run)}</td><td>{esc(predictions.get(run, ''))}</td></tr>")
    return (
        "<table class='predictions'><thead><tr><th>run</th><th>prediction</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def build_html(records: list[dict], image_root: Path | None, out_dir: Path, max_side: int, jpeg_quality: int) -> dict:
    image_manifest = []
    missing_images = []
    cards = []

    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        by_category[row["primary_error_type"]].append(row)

    toc_items = []
    for category, rows in sorted(by_category.items(), key=lambda item: (-len(item[1]), item[0])):
        toc_items.append(f"<li><a href='#{esc(category)}'>{esc(category)}</a> ({len(rows)})</li>")

    for category, rows in sorted(by_category.items(), key=lambda item: (-len(item[1]), item[0])):
        cards.append(f"<h2 id='{esc(category)}'>{esc(category)} ({len(rows)})</h2>")
        for row in rows:
            image_name = Path(row["image_path"]).name
            image_path = image_root / image_name if image_root else Path("__missing__") / image_name
            data_url, meta = image_to_data_url(image_path, max_side=max_side, jpeg_quality=jpeg_quality)
            meta["sample_index"] = row["sample_index"]
            image_manifest.append(meta)
            if data_url is None:
                missing_images.append({"sample_index": row["sample_index"], "image": image_name, "path": str(image_path)})
                image_html = (
                    "<div class='missing-image'>"
                    f"Missing image: {esc(image_name)}<br>"
                    f"Expected root: {esc(str(image_root) if image_root else 'not found')}"
                    "</div>"
                )
            else:
                image_html = (
                    f"<img class='chart' src='{data_url}' "
                    f"alt='ChartQA sample {esc(row['sample_index'])}'>"
                )

            tags = ", ".join(row.get("audit_tags", []))
            cards.append(
                "<article class='sample-card'>"
                f"<div class='sample-image'>{image_html}</div>"
                "<div class='sample-info'>"
                f"<h3>sample_index={esc(row['sample_index'])} | human_or_machine={esc(row.get('human_or_machine'))}</h3>"
                f"<p><strong>Question:</strong> {esc(row.get('question'))}</p>"
                f"<p><strong>Reference:</strong> <code>{esc(row.get('reference'))}</code></p>"
                f"<p><strong>Primary label:</strong> <code>{esc(row.get('primary_error_type'))}</code></p>"
                f"<p><strong>Audit tags:</strong> {esc(tags)}</p>"
                f"<p><strong>Consensus:</strong> {esc(row.get('consensus_note'))}</p>"
                f"<p><strong>Source image:</strong> <code>{esc(row.get('image_path'))}</code></p>"
                f"{render_prediction_table(row.get('predictions', {}))}"
                "</div>"
                "</article>"
            )

    summary = make_summary(records)
    generated = date.today().isoformat()
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>ChartQA Full-Val All-Runs-Wrong Review Pack</title>
<style>
body {{
  margin: 0;
  font-family: Arial, "Microsoft YaHei", sans-serif;
  background: #f6f7f9;
  color: #18202a;
}}
header {{
  padding: 28px 36px;
  background: #102033;
  color: white;
}}
main {{
  max-width: 1320px;
  margin: 0 auto;
  padding: 24px 24px 64px;
}}
h1, h2, h3 {{ margin-top: 0; }}
h2 {{
  border-top: 3px solid #102033;
  padding-top: 18px;
  margin-top: 36px;
}}
.context, .summary-block, .toc {{
  background: white;
  border: 1px solid #d9dee8;
  border-radius: 8px;
  padding: 18px 20px;
  margin: 16px 0;
}}
.sample-card {{
  display: grid;
  grid-template-columns: minmax(420px, 58%) minmax(360px, 42%);
  gap: 18px;
  align-items: start;
  background: white;
  border: 1px solid #d9dee8;
  border-radius: 8px;
  margin: 18px 0;
  padding: 14px;
  break-inside: avoid;
}}
.chart {{
  width: 100%;
  height: auto;
  border: 1px solid #cfd6e2;
  background: white;
}}
.missing-image {{
  min-height: 280px;
  display: flex;
  align-items: center;
  justify-content: center;
  border: 1px dashed #b34a4a;
  color: #8f2424;
  background: #fff7f7;
  text-align: center;
  padding: 16px;
}}
table {{
  border-collapse: collapse;
  width: 100%;
  margin: 8px 0 16px;
  font-size: 14px;
}}
th, td {{
  border: 1px solid #d5dbe6;
  padding: 6px 8px;
  vertical-align: top;
}}
th {{ background: #eef2f7; text-align: left; }}
code {{
  background: #eef2f7;
  padding: 1px 4px;
  border-radius: 4px;
}}
.meta {{
  opacity: 0.85;
  font-size: 14px;
}}
@media (max-width: 900px) {{
  .sample-card {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<header>
<h1>ChartQA Full-Val All-Runs-Wrong Review Pack</h1>
<p class="meta">Generated: {esc(generated)} | Records: {len(records)} | Images embedded: {len(records) - len(missing_images)} | Missing images: {len(missing_images)}</p>
</header>
<main>
<section class="context">
<h2>项目背景、进展、目标与规划</h2>
{''.join(f'<p>{esc(paragraph)}</p>' for paragraph in PROJECT_CONTEXT.splitlines() if paragraph.strip())}
</section>
<section class="summary-block">
<h2>当前审计汇总</h2>
{render_summary_tables(summary)}
</section>
<section class="toc">
<h2>目录</h2>
<ul>{''.join(toc_items)}</ul>
</section>
{''.join(cards)}
</main>
</body>
</html>
"""

    html_path = out_dir / "chartqa_full_val_all_wrong_webchat_review_pack.html"
    html_path.write_text(html_text, encoding="utf-8")

    manifest_path = out_dir / "chartqa_full_val_all_wrong_webchat_review_pack_manifest.json"
    manifest = {
        "html_path": str(html_path),
        "record_count": len(records),
        "embedded_image_count": len(records) - len(missing_images),
        "missing_image_count": len(missing_images),
        "image_root": str(image_root) if image_root else None,
        "max_side": max_side,
        "jpeg_quality": jpeg_quality,
        "summary": summary,
        "missing_images": missing_images,
        "image_manifest": image_manifest,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    return manifest


def build_markdown(records: list[dict], out_dir: Path) -> Path:
    summary = make_summary(records)
    lines = [
        "# ChartQA Full-Val All-Runs-Wrong Web Chat Review Pack",
        "",
        "## Project Context",
        "",
        PROJECT_CONTEXT,
        "",
        "## Summary",
        "",
        f"- Total samples: {summary['total']}",
        f"- High-consensus wrong samples: {summary['high_consensus_wrong_count']}",
        "",
        "### Primary Error Type Counts",
        "",
        "| primary_error_type | count |",
        "|---|---:|",
    ]
    for key, count in sorted(summary["primary_error_type_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{key}` | {count} |")

    lines.extend(["", "## Samples", ""])
    for row in records:
        lines.extend(
            [
                f"### sample_index={row['sample_index']} | {row['primary_error_type']}",
                "",
                f"- human_or_machine: `{row.get('human_or_machine')}`",
                f"- question: {row.get('question')}",
                f"- reference: `{row.get('reference')}`",
                f"- tags: `{', '.join(row.get('audit_tags', []))}`",
                f"- consensus: {row.get('consensus_note')}",
                f"- image_path: `{row.get('image_path')}`",
                "",
                "| run | prediction |",
                "|---|---|",
            ]
        )
        for run in RUN_ORDER:
            lines.append(f"| `{run}` | `{str(row.get('predictions', {}).get(run, '')).replace('|', '/')}` |")
        lines.append("")

    md_path = out_dir / "chartqa_full_val_all_wrong_webchat_review_pack.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit-json",
        default="outputs/chartqa_all_runs_wrong_audit/chartqa_full_val_all_runs_wrong_model_audit.json",
    )
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--output-dir", default="outputs/chartqa_all_runs_wrong_review_pack")
    parser.add_argument("--max-side", type=int, default=1400)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(Path(args.audit_json))
    image_root = find_image_root(args.image_root)

    manifest = build_html(
        records=records,
        image_root=image_root,
        out_dir=out_dir,
        max_side=args.max_side,
        jpeg_quality=args.jpeg_quality,
    )
    md_path = build_markdown(records, out_dir)

    print(f"HTML: {manifest['html_path']}")
    print(f"Markdown: {md_path}")
    print(f"Records: {manifest['record_count']}")
    print(f"Images embedded: {manifest['embedded_image_count']}")
    print(f"Missing images: {manifest['missing_image_count']}")
    print(f"Image root: {manifest['image_root']}")


if __name__ == "__main__":
    main()
