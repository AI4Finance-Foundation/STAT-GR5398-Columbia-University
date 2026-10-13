"""Render technical_appendix.md -> technical_appendix.pdf (python-markdown + headless Chrome). Run from this folder."""
import base64, re, subprocess, pathlib, markdown
src = pathlib.Path("technical_appendix.md").read_text()
meta, body = re.match(r"---\n(.*?)\n---\n(.*)", src, re.S).groups()
m = dict(re.findall(r'(\w+):\s*"(.*)"', meta))
# inline images so the PDF does not depend on relative paths
def inline(mo):
    data = base64.b64encode(pathlib.Path(mo.group(2)).read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{data}"><figcaption>{mo.group(1)}</figcaption></figure>'
body = re.sub(r"!\[(.*?)\]\((.*?)\)", inline, body)
html = markdown.markdown(body, extensions=["tables", "sane_lists"])
page = f"""<!doctype html><html><head><meta charset="utf-8"><title>{m['title']}</title>
<style>{pathlib.Path('style.css').read_text()}</style></head><body>
<h1 class="title">{m['title']}</h1><p class="author">{m['author']}</p><p class="date">{m['date']}</p>{html}</body></html>"""
pathlib.Path("technical_appendix.html").write_text(page)
import os, shutil
candidates = [os.environ.get("CHROME", ""), "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
              shutil.which("google-chrome") or "", shutil.which("chromium") or "", shutil.which("chromium-browser") or ""]
chrome = next((c for c in candidates if c and os.path.exists(c)), None)
if chrome is None:
    raise SystemExit("No Chrome/Chromium found; set $CHROME to its path")
subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                "--print-to-pdf=technical_appendix.pdf", f"file://{pathlib.Path('technical_appendix.html').resolve()}"], check=True,
               capture_output=True)
print("wrote technical_appendix.pdf")
