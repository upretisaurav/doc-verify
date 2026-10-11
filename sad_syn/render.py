"""Offer letters: templates, renderer and the values OCR should find on the page."""

from sad_syn.layout import Page, format_currency, format_date, format_duration

OFFER_TEMPLATES = [
    {
        "template_id": "offer_01",
        "font": "LiberationSans-Regular.ttf",
        "margin": 150,
        "header_style": "centered",
        "fee_layout": "table",
        "fees_first": False,
        "line_spacing_factor": 1.4,
        "font_size_main": 32,
        "font_size_title": 48,
        "date_format": "%B %d, %Y",
    },
    {
        "template_id": "offer_02",
        "font": "LiberationSerif-Regular.ttf",
        "margin": 170,
        "header_style": "left",
        "fee_layout": "list",
        "fees_first": False,
        "line_spacing_factor": 1.45,
        "font_size_main": 33,
        "font_size_title": 50,
        "date_format": "%d %B %Y",
    },
    {
        "template_id": "offer_03",
        "font": "CourierPrime-Regular.ttf",
        "margin": 140,
        "header_style": "centered",
        "fee_layout": "table",
        "fees_first": True,
        "line_spacing_factor": 1.4,
        "font_size_main": 30,
        "font_size_title": 44,
        "date_format": "%d/%m/%Y",
    },
]

template_01 = OFFER_TEMPLATES[0]

OFFER_PRINTED_TWICE = ('payment_deadline',)


def offer_ocr_fields(offer, template):
    currency = offer['currency']
    return {
        'university_header': offer['university_header'],
        'university_body': offer['university_body'],
        'acceptance_deposit_header': format_currency(offer['acceptance_deposit_header'], currency),
        'acceptance_deposit_body': format_currency(offer['acceptance_deposit_body'], currency),
        'first_year_tuition_fee': format_currency(offer['first_year_tuition_fee'], currency),
        'total_estimated_tuition': format_currency(offer['total_estimated_tuition'], currency),
        'date_of_issue': format_date(offer['date_of_issue'], template),
        'payment_deadline': format_date(offer['payment_deadline'], template),
        'orientation_date': format_date(offer['orientation_date'], template),
        'commencement_date': format_date(offer['commencement_date'], template),
        'completion_date': format_date(offer['completion_date'], template),
    }


def render_offer(offer, template, output_filename=None):
    page = Page(template)
    fields = offer_ocr_fields(offer, template)

    def draw_body():
        body_text = (
            f"We are pleased to offer you admission to {fields['university_body']} "
            f"in the {offer['program_title']} program.\n\n"
            f"Faculty: {offer['faculty_department']}\n"
            f"Study Mode: {offer['study_mode']} at {offer['campus_location']}\n"
            f"Duration: {format_duration(offer['duration_years'])}\n"
            f"Orientation: {fields['orientation_date']}\n"
            f"Commencement: {fields['commencement_date']}\n"
            f"Completion: {fields['completion_date']}\n\n"
            f"To accept this offer and secure your enrollment, please pay a deposit "
            f"of {fields['acceptance_deposit_body']} by {fields['payment_deadline']}."
        )
        page.paragraph(body_text)
        page.blank()

    def draw_fees():
        page.line("Financial Summary:")
        page.blank()

        fee_rows = [
            ("First-Year Tuition:", fields['first_year_tuition_fee']),
            ("Estimated Total:", fields['total_estimated_tuition']),
            ("Deposit Required:", fields['acceptance_deposit_header']),
            ("Payment Deadline:", fields['payment_deadline']),
        ]
        for label, value in fee_rows:
            if template['fee_layout'] == "table":
                page.row([label, value], [0, 600])
            else:
                page.line(f"{label} {value}")
        page.blank()

    page.header(fields['university_header'], offer['contact_info'])
    page.blank()

    page.line(f"Date: {fields['date_of_issue']}")
    page.blank()
    page.line(offer['applicant_address'])
    page.line(f"Student ID: {offer['student_id']} | App No: {offer['application_number']}")
    page.blank()
    page.line(f"Dear {offer['student_name']},")
    page.blank()

    if template['fees_first']:
        draw_fees()
        draw_body()
    else:
        draw_body()
        draw_fees()

    page.line(f"Conditions: {offer['conditions']}")
    page.blank()
    page.line("Sincerely,")
    page.blank()
    page.line(offer['signatory'])

    return page.finish(output_filename)
