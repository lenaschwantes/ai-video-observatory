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
    "pt": ("GRÁTIS · SEM CADASTRO", "Esse vídeo foi\nfeito com IA?",
           "Cole o link e veja em cerca de um minuto,\ncom as provas.", "Observatório.IA"),
    "en": ("FREE · NO SIGN-UP", "Was this video\nmade with AI?",
           "Paste the link and see in about a minute,\nwith the proof.", "Observatory.AI"),
}


BUTTON = {"pt": "Verificar", "en": "Check"}


def render(lang: str) -> None:
    kicker, headline, sub, brand = COPY[lang]
    img = Image.new("RGB", (1200, 630), PAGE)
    d = ImageDraw.Draw(img)
    d.text((80, 70), kicker, font=ImageFont.truetype(SANS_BOLD, 26), fill=ACCENT)
    d.multiline_text((80, 120), headline, font=ImageFont.truetype(SERIF, 72), fill=INK, spacing=10)
    d.multiline_text((80, 330), sub, font=ImageFont.truetype(SANS, 30), fill=INK2, spacing=8)
    # the product in one picture: a link field and the check button
    d.rounded_rectangle((80, 450, 700, 530), radius=16, fill="#ffffff", outline="#c3c2b7", width=3)
    d.text((110, 490), "youtube.com/shorts/…", font=ImageFont.truetype(SANS, 30), fill="#898781", anchor="lm")
    d.rounded_rectangle((716, 450, 960, 530), radius=16, fill=ACCENT)
    d.text((838, 490), BUTTON[lang], font=ImageFont.truetype(SANS_BOLD, 30), fill="#ffffff", anchor="mm")
    d.text((1120, 100), brand, font=ImageFont.truetype(SERIF, 34), fill=INK, anchor="rs")
    img.save(OUT / f"og-{lang}.png", optimize=True)
    print("wrote", OUT / f"og-{lang}.png")


if __name__ == "__main__":
    for lang in COPY:
        render(lang)
