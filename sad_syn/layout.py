"""Shared page-drawing and formatting helpers for every SAD-Syn document type."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONTS_DIR = Path(__file__).resolve().parents[1] / "fonts"

PAGE_WIDTH, PAGE_HEIGHT = 1654, 2339

CURRENCY_SYMBOLS = {"USD": "$", "AUD": "A$", "GBP": "£", "CAD": "C$"}


def format_currency(amount, currency):
    sym = CURRENCY_SYMBOLS.get(currency, currency + " ")
    return f"{sym}{amount:,}"


def format_cents(cents, currency=None):
    amount = f"{cents // 100:,}.{cents % 100:02d}"
    if currency is None:
        return amount
    return CURRENCY_SYMBOLS.get(currency, currency + " ") + amount


def format_duration(years):
    if years == 1.0:
        return "1 year"
    elif years.is_integer():
        return f"{int(years)} years"
    return f"{years} years"


def format_date(d, template):
    return d.strftime(template['date_format'])


def wrap_text(text, font, max_width):
    words = text.split()
    lines = []
    current_line = []
    for word in words:
        test_line = ' '.join(current_line + [word])
        if font.getlength(test_line) <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
            else:
                lines.append(word)
                current_line = []
    if current_line:
        lines.append(' '.join(current_line))
    return lines


class Page:
    """An A4 page at 200 DPI with a text cursor that moves down as lines are drawn."""

    def __init__(self, template):
        self.template = template
        self.img = Image.new('RGB', (PAGE_WIDTH, PAGE_HEIGHT), color='white')
        self.draw = ImageDraw.Draw(self.img)

        font_path = str(FONTS_DIR / template['font'])
        self.fonts = {
            'main': ImageFont.truetype(font_path, template['font_size_main']),
            'title': ImageFont.truetype(font_path, template['font_size_title']),
        }

        self.margin = template['margin']
        self.text_width = PAGE_WIDTH - (self.margin * 2)
        self.y = self.margin

    def _advance(self, font, text):
        self.y += int(font.size * self.template['line_spacing_factor'])
        if self.y > (PAGE_HEIGHT - self.margin):
            raise ValueError(f"Page overflow detected at line: '{text[:20]}...'")

    def line(self, text, font='main', align='left'):
        font = self.fonts[font]
        for line in wrap_text(text, font, self.text_width) or [""]:
            if align == 'centered':
                x = (PAGE_WIDTH - font.getlength(line)) / 2
            else:
                x = self.margin

            self.draw.text((x, self.y), line, font=font, fill='black')
            self._advance(font, line)

    def blank(self):
        self.line("")

    def paragraph(self, text):
        for p in text.split('\n'):
            self.line(p)

    def header(self, name, contact_info):
        align = 'centered' if self.template['header_style'] == 'centered' else 'left'
        self.line(name, 'title', align)
        self.line(contact_info, 'main', align)

    def columns(self, *fractions):
        return [round(fraction * self.text_width) for fraction in fractions]

    def row(self, cells, offsets):
        font = self.fonts['main']
        for cell, offset in zip(cells, offsets):
            self.draw.text((self.margin + offset, self.y), cell, font=font, fill='black')
        self._advance(font, cells[0])

    def finish(self, output_filename=None):
        if output_filename is not None:
            self.img.save(output_filename)
            print(f"Saved: {output_filename}")
        return self.img
