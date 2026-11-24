#!/usr/bin/env python3
"""Render src/posts/*.md -> blog/<slug>.html with pandoc. No other dependencies.

Usage: python3 build.py            # src/posts  -> blog/
       python3 build.py --drafts   # src/drafts -> drafts/ (gitignored, local preview only)
Post filenames: YYYY-MM-DD-slug.md, with YAML front matter (title, date, ...).
Set `katex: true` in front matter to load KaTeX on that page only.
"""
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).parent
DRAFTS = "--drafts" in sys.argv
SRC = ROOT / "src" / ("drafts" if DRAFTS else "posts")
OUT = ROOT / ("drafts" if DRAFTS else "blog")
TEMPLATE = ROOT / "src" / "post.html"


def preprocess(text: str) -> str:
    # Strip leftover Jekyll/Liquid URL prefixes from image paths.
    text = text.replace("{{ site.url }}{{ site.baseurl }}", "")
    # Kramdown ".notice" blocks: a paragraph followed by `{: .notice...}`.
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("{: .notice"):
            start = i
            while start > 0 and lines[start - 1].strip():
                start -= 1
            lines[start] = "::: notice\n" + lines[start]
            lines[i] = ":::"
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for md in sorted(SRC.glob("*.md")):
        slug = re.sub(r"^\d{4}-\d{2}-\d{2}-", "", md.stem)
        has_math = re.search(r"^katex:\s*true", md.read_text(), re.M)
        args = [
            "pandoc", "-f", "markdown+tex_math_dollars", "-t", "html5",
            "--template", str(TEMPLATE), "--no-highlight",
            "-o", str(OUT / f"{slug}.html"),
        ]
        if has_math:
            args.append("--katex")
        subprocess.run(args, input=preprocess(md.read_text()), text=True, check=True)
        out = OUT / f"{slug}.html"
        out.write_text(out.read_text().replace("<img ", '<img loading="lazy" '))
        print(f"built {OUT.name}/{slug}.html")


if __name__ == "__main__":
    main()
