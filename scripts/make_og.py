"""Render the link-preview images (1200x630) shown when the site is shared
on LinkedIn, WhatsApp or X. Needs Pillow and the Georgia/Segoe UI fonts
(Windows) or pass other TTF paths via OG_SERIF / OG_SANS.

Run: python scripts/make_og.py
"""

import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parents[1] / "src" / "observatory" / "detect" / "static"
FONTS = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
SERIF = os.environ.get("OG_SERIF", str(FONTS / "georgiab.ttf"))
SANS = os.environ.get("OG_SANS", str(FONTS / "segoeui.ttf"))
SANS_BOLD = os.environ.get("OG_SANS_BOLD", str(FONTS / "segoeuib.ttf"))

PAGE, INK, INK2, ACCENT, AI, HUMAN = "#f9f9f7", "#0b0b0b", "#52514e", "#c2410c", "#e34948", "#2a78d6"

COPY = {
    "pt": ("TESTE INDEPENDENTE", "Detectores de vídeo de IA\nacusam vídeos reais?",
           "Ferramentas comerciais e modelos de IA testados nos mesmos vídeos,\nincluindo conteúdo brasileiro.", "Observatório.IA"),
    "en": ("INDEPENDENT TEST", "Do AI-video detectors\naccuse real videos?",
           "Commercial tools and AI models tested on the same videos,\nBrazilian content included.", "Observatory.AI"),
}


def render(lang: str) -> None:
    kicker, headline, sub, brand = COPY[lang]
    img = Image.new("RGB", (1200, 630), PAGE)
    d = ImageDraw.Draw(img)
    d.text((80, 70), kicker, font=ImageFont.truetype(SANS_BOLD, 26), fill=ACCENT)
    d.multiline_text((80, 120), headline, font=ImageFont.truetype(SERIF, 72), fill=INK, spacing=10)
    d.multiline_text((80, 330), sub, font=ImageFont.truetype(SANS, 30), fill=INK2, spacing=8)
    # a small ranking motif: bars of decreasing length, the top one in the "accused" colour
    for i, (w, colour) in enumerate([(520, AI), (430, AI), (300, AI), (170, HUMAN)]):
        y = 450 + i * 34
        d.rounded_rectangle((80, y, 80 + w, y + 20), radius=6, fill=colour)
    d.text((1120, 560), brand, font=ImageFont.truetype(SERIF, 34), fill=INK, anchor="rs")
    img.save(OUT / f"og-{lang}.png", optimize=True)
    print("wrote", OUT / f"og-{lang}.png")


if __name__ == "__main__":
    for lang in COPY:
        render(lang)
