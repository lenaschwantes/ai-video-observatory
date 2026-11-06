"""Render the link-preview images (1200x630) shown when the site is shared
on LinkedIn, WhatsApp or X: the question on the left, a real result on the right.

Needs Pillow. Headings use Fraunces (the site's heading font): pass the TTF via
OG_SERIF, e.g. the variable font from github.com/google/fonts (ofl/fraunces).
Without it, falls back to Georgia Bold. Body text uses Segoe UI (Windows) or
OG_SANS / OG_SANS_BOLD.

Run: OG_SERIF=path/to/Fraunces.ttf python scripts/make_og.py
"""

import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src" / "observatory" / "detect" / "static"
FRAME = ROOT / "examples" / "reports" / "youtube-weizh2jfsnw" / "frame_06.jpg"
FONTS = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
SERIF = os.environ.get("OG_SERIF", str(FONTS / "georgiab.ttf"))
SANS = os.environ.get("OG_SANS", str(FONTS / "segoeui.ttf"))
SANS_BOLD = os.environ.get("OG_SANS_BOLD", str(FONTS / "segoeuib.ttf"))

PAGE, INK, INK2, MUTED, RING = "#f9f9f7", "#0b0b0b", "#52514e", "#898781", "#dcdbd3"
ACCENT, AI, AI_WASH, MID, HUMAN_WASH = "#c2410c", "#e34948", "#fbe4e3", "#ecebe6", "#e3eefb"

COPY = {
    "pt": {"kicker": "GRÁTIS · SEM CADASTRO", "headline": "Esse vídeo foi\nfeito com IA?",
           "sub": "Cole o link e veja em cerca de\num minuto, com as provas.", "brand": "Observatório.IA",
           "paste": "Cole o link do vídeo", "button": "Verificar", "label": "EXEMPLO REAL",
           "verdict": "Provavelmente feito\ncom IA",
           "clue": "Texto do relógio sai amassado"},
    "en": {"kicker": "FREE · NO SIGN-UP", "headline": "Was this video\nmade with AI?",
           "sub": "Paste the link and see in about\na minute, with the proof.", "brand": "Observatory.AI",
           "paste": "Paste the video link", "button": "Check", "label": "REAL EXAMPLE",
           "verdict": "Likely made\nwith AI",
           "clue": "Text on the clock is squashed"},
}


def serif(size: int) -> ImageFont.FreeTypeFont:
    font = ImageFont.truetype(SERIF, size)
    try:  # variable Fraunces: pick the weight the site uses
        font.set_variation_by_name("SemiBold")
    except (OSError, ValueError, AttributeError):
        pass
    return font


def sans(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(SANS_BOLD if bold else SANS, size)


def background() -> Image.Image:
    img = Image.new("RGB", (1200, 630), PAGE)
    glow = Image.new("L", (1200, 630), 0)
    ImageDraw.Draw(glow).ellipse((620, -260, 1420, 520), fill=90)
    glow = glow.filter(ImageFilter.GaussianBlur(120))
    return Image.composite(Image.new("RGB", img.size, "#f6c9b0"), img, glow)


def card(c: dict) -> Image.Image:
    """The example result, tilted slightly like on the home page."""
    w, h = 400, 450
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, w - 1, h - 1), radius=22, fill="#ffffff", outline=RING, width=2)
    d.text((22, 24), c["label"], font=sans(15, True), fill=MUTED, anchor="lm")
    d.line((0, 48, w, 48), fill=RING, width=2)
    frame = ImageOps.fit(Image.open(FRAME).convert("RGB"), (w - 4, 216))
    im.paste(frame, (2, 50))
    d.rounded_rectangle((w - 118, 196, w - 18, 250), radius=12, fill=AI)
    d.text((w - 68, 223), "84%", font=sans(32, True), fill="#ffffff", anchor="mm")
    y = 286
    d.multiline_text((22, y), c["verdict"], font=serif(30), fill=INK, spacing=4)
    by = y + 92
    d.rounded_rectangle((22, by, 22 + 89, by + 8), radius=4, fill=HUMAN_WASH)
    d.rectangle((113, by, 113 + 176, by + 8), fill=MID)
    d.rounded_rectangle((291, by, w - 22, by + 8), radius=4, fill=AI_WASH)
    mx = 22 + int(0.84 * (w - 44))
    d.rounded_rectangle((mx - 2, by - 6, mx + 2, by + 14), radius=2, fill=INK)
    d.rounded_rectangle((22, by + 30, 82, by + 56), radius=7, fill=AI_WASH)
    d.text((52, by + 43), "21.5s", font=sans(15, True), fill=AI, anchor="mm")
    d.text((94, by + 43), c["clue"], font=sans(19), fill=INK2, anchor="lm")
    return im


def render(lang: str) -> None:
    c = COPY[lang]
    img = background()
    d = ImageDraw.Draw(img)
    d.text((72, 64), c["brand"], font=serif(30), fill=INK, anchor="ls")
    kf = sans(20, True)
    kw = d.textlength(c["kicker"], font=kf) + 32
    d.rounded_rectangle((72, 100, 72 + kw, 138), radius=19, fill="#ffffff", outline=RING, width=2)
    d.text((88, 119), c["kicker"], font=kf, fill=ACCENT, anchor="lm")
    d.multiline_text((72, 160), c["headline"], font=serif(76), fill=INK, spacing=8)
    d.multiline_text((72, 352), c["sub"], font=sans(28), fill=INK2, spacing=6)
    # the product in one picture: a link field and the check button
    d.rounded_rectangle((72, 470, 470, 546), radius=16, fill="#ffffff", outline="#c3c2b7", width=3)
    d.text((98, 508), c["paste"], font=sans(26), fill=MUTED, anchor="lm")
    d.rounded_rectangle((484, 470, 650, 546), radius=16, fill=ACCENT)
    d.text((567, 508), c["button"], font=sans(26, True), fill="#ffffff", anchor="mm")

    k = card(c)
    shadow = Image.new("RGBA", (k.width + 80, k.height + 80), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle((46, 56, k.width + 34, k.height + 46), radius=22, fill=(0, 0, 0, 55))
    shadow = shadow.filter(ImageFilter.GaussianBlur(22)).rotate(-2, resample=Image.BICUBIC, expand=True)
    tilted = k.rotate(-2, resample=Image.BICUBIC, expand=True)
    img.paste(shadow, (722 - 40, 80 - 40), shadow)
    img.paste(tilted, (722, 80), tilted)
    img.save(OUT / f"og-{lang}.png", optimize=True)
    print("wrote", OUT / f"og-{lang}.png")


if __name__ == "__main__":
    for lang in COPY:
        render(lang)
