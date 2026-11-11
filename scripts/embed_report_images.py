"""Make a self-contained copy of a Markdown report for PDF export: every image link to a local file is replaced by
the image itself, embedded as JPEG (base64, reference style so the text stays readable). The report in docs/ keeps
plain links and is the one committed; the copy holds images from the dataset, so it is written under data/ (not
committed). Text is copied unchanged apart from the image links.

  python scripts/embed_report_images.py [--src docs/BAO_CAO_TIEN_DO.md]
                                        [--out data/weedsgalore/BAO_CAO_TIEN_DO_co_hinh.md]
"""
import argparse
import base64
import io
import re
from pathlib import Path

from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
LINK = re.compile(r'!\[([^\]]*)\]\(([^)\s]+\.(?:png|jpe?g))\)')


def embed(text, base_dir, quality=92):
    """`text` with each image link ![alt](path) to a local file replaced by ![alt][fig-N], and the definitions
    [fig-N]: data:image/jpeg;base64,... appended at the end. Returns the new text and the files embedded."""
    defs, used = [], []

    def repl(m):
        path = (base_dir / m.group(2)).resolve()
        if not path.exists():
            raise FileNotFoundError(f'image not found: {m.group(2)} (resolved to {path})')
        buf = io.BytesIO()
        Image.open(path).convert('RGB').save(buf, 'JPEG', quality=quality, optimize=True)
        ref = f'fig-{len(defs) + 1}'
        defs.append(f'[{ref}]: data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}')
        used.append(path)
        return f'![{m.group(1)}][{ref}]'

    out = LINK.sub(repl, text)
    if defs:
        out = out.rstrip('\n') + '\n\n<!-- Hình nhúng sẵn (JPEG, base64), sinh bởi scripts/embed_report_images.py -->\n\n'
        out += '\n\n'.join(defs) + '\n'
    return out, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=str(ROOT_DIR / 'docs/BAO_CAO_TIEN_DO.md'))
    ap.add_argument('--out', default=str(ROOT_DIR / 'data/weedsgalore/BAO_CAO_TIEN_DO_co_hinh.md'))
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    text, used = embed(src.read_text(encoding='utf-8'), src.parent)
    out.write_bytes(text.encode('utf-8'))  # bytes: keep LF line endings on Windows
    print(f'wrote {out} ({out.stat().st_size / 1e6:.2f} MB), embedded: {[p.name for p in used]}')


if __name__ == '__main__':
    main()
