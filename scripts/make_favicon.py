"""Render the site icon: a play button inside scan brackets ("check a video").

Writes favicon.svg (adapts to dark mode), favicon.ico (16/32/48),
icon-192.png, icon-512.png and apple-touch-icon.png into the static folder.
The PNGs are drawn at 8x and scaled down, so the edges stay clean.

Run: python scripts/make_favicon.py
"""

from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parents[1] / "src" / "observatory" / "detect" / "static"
INK, PAGE, ACCENT = "#0b0b0b", "#f9f9f7", "#e5571f"

# geometry on a 64-unit grid, shared by the SVG and the PNGs
RADIUS = 15
BRACKET = [((13, 23), (13, 13), (23, 13)), ((41, 13), (51, 13), (51, 23)),
           ((51, 41), (51, 51), (41, 51)), ((23, 51), (13, 51), (13, 41))]
STROKE = 5
PLAY = [(26, 21.5), (26, 42.5), (43.5, 32)]

SVG = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <style>
    .bg {{ fill: {INK}; }} .br {{ fill: none; stroke: {PAGE}; }} .pl {{ fill: {ACCENT}; stroke: {ACCENT}; }}
    @media (prefers-color-scheme: dark) {{ .bg {{ fill: {PAGE}; }} .br {{ stroke: {INK}; }} }}
  </style>
  <rect class="bg" width="64" height="64" rx="{RADIUS}"/>
  <path class="br" stroke-width="{STROKE}" stroke-linecap="round" stroke-linejoin="round" d="{' '.join('M' + ' L'.join(f'{x} {y}' for x, y in b) for b in BRACKET)}"/>
  <path class="pl" stroke-width="4" stroke-linejoin="round" d="M{' L'.join(f'{x} {y}' for x, y in PLAY)} Z"/>
</svg>
"""


def draw(size: int, padded: bool = False) -> Image.Image:
    """padded=True keeps the mark inside the safe zone of maskable/app icons."""
    k = 8
    s = size * k
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    u = s / 64
    if padded:
        d.rectangle((0, 0, s, s), fill=INK)
        u *= 0.8
    off = (s - 64 * u) / 2

    def p(x: float, y: float) -> tuple[float, float]:
        return off + x * u, off + y * u

    if not padded:
        d.rounded_rectangle((0, 0, s - 1, s - 1), radius=RADIUS * u, fill=INK)
    w = STROKE * u
    for b in BRACKET:
        pts = [p(*q) for q in b]
        d.line(pts, fill=PAGE, width=round(w), joint="curve")
        for x, y in (pts[0], pts[-1]):
            d.ellipse((x - w / 2, y - w / 2, x + w / 2, y + w / 2), fill=PAGE)
    tri = [p(*q) for q in PLAY]
    d.polygon(tri, fill=ACCENT)
    d.line(tri + [tri[0]], fill=ACCENT, width=round(4 * u), joint="curve")
    for x, y in tri:
        r = 2 * u
        d.ellipse((x - r, y - r, x + r, y + r), fill=ACCENT)
    return img.resize((size, size), Image.LANCZOS)


def main() -> None:
    (OUT / "favicon.svg").write_text(SVG, encoding="utf-8")
    draw(48).save(OUT / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48)])
    draw(192).save(OUT / "icon-192.png", optimize=True)
    draw(512).save(OUT / "icon-512.png", optimize=True)
    draw(180, padded=True).convert("RGB").save(OUT / "apple-touch-icon.png", optimize=True)
    print("wrote favicon.svg, favicon.ico, icon-192.png, icon-512.png, apple-touch-icon.png")


if __name__ == "__main__":
    main()
