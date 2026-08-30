"""Extract each PDF in kb/literature/*/ to kb/literature/text/<name>.md so cards can quote sections."""
import sys, pathlib
from pypdf import PdfReader
root = pathlib.Path(__file__).parent; out = root / 'text'; out.mkdir(exist_ok=True)
for pdf in sorted(root.glob('*/*.pdf')):
    dst = out / (pdf.stem + '.md')
    if dst.exists(): continue
    try:
        pages = [p.extract_text() or '' for p in PdfReader(str(pdf)).pages]
    except Exception as e:
        print(f'FAIL {pdf.name}: {e}'); continue
    dst.write_text(f'# {pdf.stem}\n\n(source: {pdf.relative_to(root)}, {len(pages)} pages)\n\n' +
                   '\n\n'.join(f'<!-- page {i+1} -->\n{t}' for i, t in enumerate(pages)))
    print(f'OK   {dst.name}: {len(pages)} pages, {sum(len(t) for t in pages)//1000} KB')
