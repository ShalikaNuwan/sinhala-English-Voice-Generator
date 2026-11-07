"""Render check stills for an explainer spec into one contact sheet.
   FILM=<name> .venv/bin/python stills_tool.py <spec.json> <out_sheet.jpg> t1 t2 t3 ..."""
import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from film import D  # noqa: E402
import explainer
from PIL import Image, ImageDraw
spec = json.loads(Path(sys.argv[1]).read_text())
times = [float(x) for x in sys.argv[3:]]
with tempfile.TemporaryDirectory() as tmp:
    files = explainer.stills(spec, D, times, tmp)
    ims = [Image.open(f).resize((640, 360)) for f in files]
    cols = 3
    sheet = Image.new("RGB", (640 * cols, 360 * ((len(ims) + cols - 1) // cols)))
    for i, (im, t) in enumerate(zip(ims, times)):
        ImageDraw.Draw(im).text((8, 8), f"t={t}", fill=(255, 255, 0))
        sheet.paste(im, ((i % cols) * 640, (i // cols) * 360))
    sheet.save(sys.argv[2], quality=88)
print(sys.argv[2])
