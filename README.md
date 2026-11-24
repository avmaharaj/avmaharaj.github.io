# avmaharaj.github.io

Plain static HTML personal site. No framework and no JavaScript on the homepage. `.nojekyll` turns off GitHub Pages' Jekyll build.

- `index.html`: the homepage. Edit it directly.
- `style.css`: the only stylesheet. It handles light and dark mode automatically.
- `src/posts/*.md`: blog post sources (`YYYY-MM-DD-slug.md` with YAML front matter).
- `blog/*.html`: generated posts. Commit these.

## Writing a post

```bash
python3 build.py        # needs pandoc (brew install pandoc)
```

Add `katex: true` to a post's front matter to load KaTeX on that page only. Then add a line for the post under "Writing" in `index.html`.

## Preview locally

```bash
python3 -m http.server 8765
```
