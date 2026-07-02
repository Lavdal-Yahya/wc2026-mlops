"""Render REPORT.md → self-contained HTML with embedded base64 figures.

Output is a single .html the reader can open in any browser and print to PDF
(Ctrl+P → "Save as PDF"). No JS, no external assets, no server.
"""
from __future__ import annotations

import base64
import mimetypes
import re
from pathlib import Path

import markdown
import yaml


CSS = """
@page { size: A4; margin: 18mm; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font-family: 'Charter', 'Georgia', 'Liberation Serif', serif;
       max-width: 760px; margin: 2rem auto; padding: 0 1.2rem; color: #222; line-height: 1.55; }
h1 { font-size: 1.9rem; border-bottom: 2px solid #2a6f97; padding-bottom: 0.4rem; margin-top: 2rem; }
h2 { font-size: 1.35rem; color: #2a6f97; margin-top: 1.6rem; }
h3 { font-size: 1.1rem; color: #333; }
h1.title { text-align: center; border: none; font-size: 2.2rem; margin-top: 0; padding-top: 1.5rem; }
.subtitle { text-align: center; color: #555; font-size: 1.1rem; font-style: italic; margin-bottom: 0.6rem; }
.meta { text-align: center; color: #666; margin-bottom: 2rem; }
table { border-collapse: collapse; margin: 1rem 0; font-size: 0.93rem; width: 100%; }
th, td { border: 1px solid #ccc; padding: 0.4rem 0.7rem; text-align: left; }
th { background: #f1f5f9; font-weight: 600; }
tr:nth-child(even) td { background: #fafafa; }
code { background: #f4f4f4; padding: 0.1rem 0.35rem; border-radius: 3px; font-size: 0.9em; }
pre { background: #f4f4f4; padding: 0.8rem; border-radius: 4px; overflow-x: auto; font-size: 0.88rem; }
pre code { background: none; padding: 0; }
blockquote { border-left: 4px solid #2a6f97; margin: 1rem 0; padding: 0.3rem 1rem; color: #444; background: #f8fbfd; }
img { max-width: 100%; display: block; margin: 1rem auto; }
figcaption { text-align: center; font-size: 0.85rem; color: #666; font-style: italic; margin-top: -0.5rem; margin-bottom: 1rem; }
hr { border: none; border-top: 1px solid #ddd; margin: 2rem 0; }
ul, ol { padding-left: 1.5rem; }
a { color: #2a6f97; }
.toc { background: #f8fbfd; border: 1px solid #d6e3ec; padding: 0.6rem 1.2rem; border-radius: 4px; margin: 1.5rem 0; }
.toc ul { padding-left: 1.2rem; margin: 0.3rem 0; }
.toc > ul { padding-left: 0; }
@media print {
  body { margin: 0; max-width: none; }
  h1 { page-break-before: auto; }
  h1.title { page-break-before: avoid; }
  table, figure, img { page-break-inside: avoid; }
}
"""


def _strip_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    meta = yaml.safe_load(parts[1]) or {}
    return meta, parts[2].lstrip("\n")


_IMG_RE = re.compile(r'!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)(?:\{[^}]*\})?')


def _embed_images(md_text: str, base_dir: Path) -> str:
    """Replace ![alt](path){attrs} with ![alt](data:URI) (inlined)."""

    def repl(match: re.Match) -> str:
        alt = match.group(1)
        src = match.group(2)
        if src.startswith(("http://", "https://", "data:")):
            return match.group(0)
        img_path = (base_dir / src).resolve()
        if not img_path.exists():
            return match.group(0)
        mime, _ = mimetypes.guess_type(img_path.name)
        mime = mime or "image/png"
        b64 = base64.b64encode(img_path.read_bytes()).decode()
        return f"![{alt}](data:{mime};base64,{b64})"

    return _IMG_RE.sub(repl, md_text)


def render(md_path: Path, out_path: Path) -> None:
    raw = md_path.read_text(encoding="utf-8")
    meta, body = _strip_frontmatter(raw)
    body = _embed_images(body, md_path.parent)

    md = markdown.Markdown(
        extensions=["extra", "tables", "fenced_code", "toc", "attr_list", "smarty"],
        extension_configs={"toc": {"toc_depth": "2-3", "title": "Contents"}},
    )
    html_body = md.convert(body)

    title = meta.get("title", md_path.stem)
    subtitle = meta.get("subtitle", "")
    author = meta.get("author", "")
    date = meta.get("date", "")

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>{CSS}</style>
</head>
<body>
<h1 class="title">{title}</h1>
{f'<p class="subtitle">{subtitle}</p>' if subtitle else ''}
<p class="meta">{author} &middot; {date}</p>
{html_body}
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")
    size_kb = out_path.stat().st_size / 1024
    print(f"Wrote {out_path}  ({size_kb:,.0f} KB)")


if __name__ == "__main__":
    from src.utils.config import PROJECT_ROOT
    md = PROJECT_ROOT / "reports" / "REPORT.md"
    out = PROJECT_ROOT / "reports" / "25239_projet.html"
    render(md, out)
