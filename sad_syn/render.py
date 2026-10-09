"""Render offer dicts from sad_syn.generate into document images."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONTS_DIR = Path(__file__).resolve().parents[1] / "fonts"

PAGE_WIDTH, PAGE_HEIGHT = 1654, 2339

template_01 = {
    "template_id": "offer_01",
    "font": "LiberationSans-Regular.ttf",
    "margin": 150,
    "header_style": "centered",
    "fee_layout": "table",
    "line_spacing_factor": 1.4,
    "font_size_main": 32,
    "font_size_title": 48
}


def format_currency(amount, currency):
    symbols = {"USD": "$", "AUD": "A$", "GBP": "£", "CAD": "C$"}
    sym = symbols.get(currency, currency + " ")
    return f"{sym}{amount:,}"


def format_duration(years):
    if years == 1.0:
        return "1 year"
    elif years.is_integer():
        return f"{int(years)} years"
    return f"{years} years"


def format_date(d):
    return d.strftime('%B %d, %Y')


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


def render_offer(offer, template, output_filename=None):
    page_width, page_height = PAGE_WIDTH, PAGE_HEIGHT
    img = Image.new('RGB', (page_width, page_height), color='white')
    draw = ImageDraw.Draw(img)

    font_path = str(FONTS_DIR / template['font'])
    font_main = ImageFont.truetype(font_path, template['font_size_main'])
    font_title = ImageFont.truetype(font_path, template['font_size_title'])

    margin = template['margin']
    max_text_width = page_width - (margin * 2)
    y_cursor = margin

    def draw_line(text, font, align="left"):
        nonlocal y_cursor
        for line in wrap_text(text, font, max_text_width) or [""]:
            if align == "centered":
                x = (page_width - font.getlength(line)) / 2
            else:
                x = margin

            draw.text((x, y_cursor), line, font=font, fill='black')
            y_cursor += int(font.size * template['line_spacing_factor'])

            if y_cursor > (page_height - margin):
                raise ValueError(f"Page overflow detected at line: '{line[:20]}...'")

    def draw_paragraph(text, font):
        for p in text.split('\n'):
            draw_line(p, font)

    if template['header_style'] == "centered":
        draw_line(offer['university_header'], font_title, align="centered")
        draw_line(offer['contact_info'], font_main, align="centered")
    else:
        draw_line(offer['university_header'], font_title)
        draw_line(offer['contact_info'], font_main)

    draw_line("", font_main)

    draw_line(f"Date: {format_date(offer['date_of_issue'])}", font_main)
    draw_line("", font_main)
    draw_line(offer['applicant_address'], font_main)
    draw_line(f"Student ID: {offer['student_id']} | App No: {offer['application_number']}", font_main)
    draw_line("", font_main)
    draw_line(f"Dear {offer['student_name']},", font_main)
    draw_line("", font_main)

    duration_str = format_duration(offer['duration_years'])

    deposit_body_str = format_currency(offer['acceptance_deposit_body'], offer['currency'])

    body_text = (
        f"We are pleased to offer you admission to {offer['university_body']} "
        f"in the {offer['program_title']} program.\n\n"
        f"Faculty: {offer['faculty_department']}\n"
        f"Study Mode: {offer['study_mode']} at {offer['campus_location']}\n"
        f"Duration: {duration_str}\n"
        f"Orientation: {format_date(offer['orientation_date'])}\n"
        f"Commencement: {format_date(offer['commencement_date'])}\n"
        f"Completion: {format_date(offer['completion_date'])}\n\n"
        f"To accept this offer and secure your enrollment, please pay a deposit "
        f"of {deposit_body_str} by {format_date(offer['payment_deadline'])}."
    )
    draw_paragraph(body_text, font_main)

    draw_line("", font_main)

    draw_line("Financial Summary:", font_main)
    draw_line("", font_main)

    if template['fee_layout'] == "table":
        def draw_table_row(col1, col2):
            nonlocal y_cursor
            draw.text((margin, y_cursor), col1, font=font_main, fill='black')
            draw.text((margin + 600, y_cursor), col2, font=font_main, fill='black')
            y_cursor += int(font_main.size * template['line_spacing_factor'])
            if y_cursor > (page_height - margin):
                raise ValueError("Page overflow detected in table layout.")

        deposit_header_str = format_currency(offer['acceptance_deposit_header'], offer['currency'])
        first_year_str = format_currency(offer['first_year_tuition_fee'], offer['currency'])
        total_str = format_currency(offer['total_estimated_tuition'], offer['currency'])

        draw_table_row("First-Year Tuition:", first_year_str)
        draw_table_row("Estimated Total:", total_str)
        draw_table_row("Deposit Required:", deposit_header_str)
        draw_table_row("Payment Deadline:", format_date(offer['payment_deadline']))

    draw_line("", font_main)

    draw_line(f"Conditions: {offer['conditions']}", font_main)
    draw_line("", font_main)
    draw_line("Sincerely,", font_main)
    draw_line("", font_main)
    draw_line(offer['signatory'], font_main)

    if output_filename is not None:
        img.save(output_filename)
        print(f"Saved: {output_filename}")

    return img
