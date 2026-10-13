"""Academic transcripts: generator, consistency rules, templates, renderer and OCR values."""

import datetime
import random

from sad_syn.generate import COUNTRIES, FIXED_REFERENCE_DATE, fake, make_university_name
from sad_syn.layout import Page, format_date

COURSES = {
    "Computer Science": ("COMP", [
        "Algorithms and Data Structures", "Operating Systems", "Database Systems", "Computer Networks",
        "Software Engineering", "Machine Learning", "Computer Vision", "Distributed Systems",
        "Programming Languages", "Cyber Security", "Human Computer Interaction", "Research Project",
    ]),
    "Data Analytics": ("DATA", [
        "Statistical Modelling", "Data Visualisation", "Big Data Platforms", "Applied Machine Learning",
        "Data Mining", "Time Series Analysis", "Database Design", "Predictive Analytics",
        "Data Ethics and Privacy", "Probability Theory", "Business Intelligence", "Analytics Project",
    ]),
    "Business Administration": ("BUSN", [
        "Financial Accounting", "Marketing Management", "Organisational Behaviour", "Business Strategy",
        "Operations Management", "Corporate Finance", "Managerial Economics", "Business Law",
        "Leadership and Ethics", "International Business", "Entrepreneurship", "Consulting Project",
    ]),
    "Engineering": ("ENGR", [
        "Engineering Mathematics", "Thermodynamics", "Fluid Mechanics", "Control Systems",
        "Materials Science", "Structural Analysis", "Circuit Theory", "Engineering Design",
        "Project Management", "Signals and Systems", "Sustainable Engineering", "Capstone Project",
    ]),
    "Design": ("DSGN", [
        "Design Thinking", "Visual Communication", "Typography", "Interaction Design",
        "Design History", "Digital Illustration", "User Experience Research", "Product Design Studio",
        "Motion Graphics", "Design for Sustainability", "Branding and Identity", "Portfolio Project",
    ]),
}

# one page holds a short coursework program: a one-year diploma or an 18-month master's
PROGRAMS = {
    "Graduate Diploma": (1.0, 8),
    "Master's": (1.5, 12),
}
COURSE_CREDITS = [3, 3, 6]
GRADE_BANDS = [(85, "HD"), (75, "D"), (65, "C"), (50, "P")]

TRANSCRIPT_PRINTED_TWICE = ()

TRANSCRIPT_EXPECTED_BROKEN_RULE = {
    "altered_date": "date_order",
    "altered_amount": "wams_match",
    "institution_mismatch": "university_names_match",
    "total_mismatch": "credits_match_courses",
}


def grade_for(mark):
    return next(grade for minimum, grade in GRADE_BANDS if mark >= minimum)


def weighted_average_mark(credits, marks):
    return round(sum(c * m for c, m in zip(credits, marks)) / sum(credits), 1)


def draw_credits(n_courses):
    return [random.choice(COURSE_CREDITS) for _ in range(n_courses)]


def draw_marks(n_courses):
    return [random.randint(50, 95) for _ in range(n_courses)]


def make_clean_transcript():
    doc = {'doc_type': 'transcript'}

    country = random.choice(list(COUNTRIES))
    local_fake = COUNTRIES[country]['faker']

    start_date_obj = FIXED_REFERENCE_DATE - datetime.timedelta(days=365)
    issue_date = fake.date_between(start_date=start_date_obj, end_date=FIXED_REFERENCE_DATE)
    completion_date = issue_date - datetime.timedelta(days=random.randint(14, 120))

    degree_level = random.choice(list(PROGRAMS))
    duration_years, n_courses = PROGRAMS[degree_level]
    commencement_date = completion_date - datetime.timedelta(days=int(duration_years * 365))

    program_subject = random.choice(list(COURSES))
    prefix, titles = COURSES[program_subject]
    if degree_level == "Graduate Diploma":
        program_title = f"Graduate Diploma in {program_subject}"
    else:
        program_title = f"Master of {program_subject}"

    credits = draw_credits(n_courses)
    marks = draw_marks(n_courses)
    courses = [
        {
            'code': f"{prefix} {random.randint(1000, 7999)}",
            'title': title,
            'credits': course_credits,
            'mark': mark,
            'grade': grade_for(mark),
        }
        for title, course_credits, mark in zip(random.sample(titles, n_courses), credits, marks)
    ]

    univ_name = make_university_name(country)
    wam = weighted_average_mark(credits, marks)

    doc['university_header'] = univ_name
    doc['university_body'] = univ_name
    doc['country'] = country
    doc['contact_info'] = local_fake.address().replace('\n', ', ')
    doc['date_of_issue'] = issue_date
    doc['student_name'] = fake.name()
    doc['student_id'] = str(fake.random_number(digits=8, fix_len=True))
    doc['program_title'] = program_title
    doc['degree_level'] = degree_level
    doc['commencement_date'] = commencement_date
    doc['completion_date'] = completion_date
    doc['courses'] = courses
    doc['total_credits'] = sum(credits)
    doc['wam_summary'] = wam
    doc['wam_statement'] = wam
    doc['signatory'] = local_fake.name() + ", Academic Registrar"

    return doc


def check_transcript_rules(doc):
    rules = {}

    rules['date_order'] = doc['commencement_date'] < doc['completion_date'] < doc['date_of_issue']

    rules['wams_match'] = (doc['wam_summary'] == doc['wam_statement'])

    rules['university_names_match'] = (doc['university_header'] == doc['university_body'])

    rules['credits_match_courses'] = (doc['total_credits'] == sum(course['credits'] for course in doc['courses']))

    return rules


def make_inconsistent_transcript(kind):
    doc = make_clean_transcript()
    n_courses = len(doc['courses'])

    if kind == "altered_date":
        doc['altered_field'] = 'completion_date'
        doc['original_value'] = doc['completion_date']

        gap_days = random.randint(1, 45)

        if random.choice([True, False]):
            doc['completion_date'] = doc['commencement_date'] - datetime.timedelta(days=gap_days)
        else:
            doc['completion_date'] = doc['date_of_issue'] + datetime.timedelta(days=gap_days)

    elif kind == "altered_amount":
        field_to_alter = random.choice(['wam_summary', 'wam_statement'])

        doc['altered_field'] = field_to_alter
        doc['original_value'] = doc[field_to_alter]

        credits = [course['credits'] for course in doc['courses']]
        new_wam = doc[field_to_alter]
        while new_wam == doc[field_to_alter]:
            new_wam = weighted_average_mark(credits, draw_marks(n_courses))

        doc[field_to_alter] = new_wam

    elif kind == "institution_mismatch":
        doc['altered_field'] = 'university_body'
        doc['original_value'] = doc['university_body']

        new_name = doc['university_body']
        while new_name == doc['university_body']:
            new_name = make_university_name(doc['country'])

        doc['university_body'] = new_name

    elif kind == "total_mismatch":
        doc['altered_field'] = 'total_credits'
        doc['original_value'] = doc['total_credits']

        wrong_total = doc['total_credits']
        while wrong_total == doc['total_credits']:
            wrong_total = sum(draw_credits(n_courses))

        doc['total_credits'] = wrong_total

    doc['inconsistency_type'] = kind

    return doc


def transcript_ocr_fields(doc, template):
    return {
        'university_header': doc['university_header'],
        'university_body': doc['university_body'],
        'wam_summary': f"{doc['wam_summary']:.1f}",
        'wam_statement': f"{doc['wam_statement']:.1f}",
        'total_credits': f"{doc['total_credits']} units",
        'commencement_date': format_date(doc['commencement_date'], template),
        'completion_date': format_date(doc['completion_date'], template),
        'date_of_issue': format_date(doc['date_of_issue'], template),
    }


def render_transcript(doc, template, output_filename=None):
    page = Page(template)
    fields = transcript_ocr_fields(doc, template)

    def draw_summary():
        page.line(f"Total credits: {fields['total_credits']}")
        page.line(f"Weighted average mark: {fields['wam_summary']}")
        page.blank()

    def draw_courses():
        offsets = page.columns(0, 0.14, 0.66, 0.78, 0.90)
        page.row(["Code", "Course", "Credits", "Mark", "Grade"], offsets)
        for course in doc['courses']:
            page.row(
                [course['code'], course['title'], str(course['credits']), str(course['mark']), course['grade']],
                offsets,
            )
        page.blank()

    page.header(fields['university_header'], doc['contact_info'])
    page.blank()
    page.line("Official Academic Transcript", align=template['title_align'])
    page.blank()

    page.line(f"Student: {doc['student_name']}")
    page.line(f"Student ID: {doc['student_id']}")
    page.line(f"Program: {doc['program_title']}")
    page.line(f"Commenced: {fields['commencement_date']}")
    page.line(f"Completed: {fields['completion_date']}")
    page.line(f"Date of issue: {fields['date_of_issue']}")
    page.blank()

    if template['summary_first']:
        draw_summary()
        draw_courses()
    else:
        draw_courses()
        draw_summary()

    page.paragraph(
        f"This is to certify that {doc['student_name']} has completed the requirements of the "
        f"{doc['program_title']} at {fields['university_body']} with a weighted average mark "
        f"of {fields['wam_statement']}."
    )
    page.blank()
    page.line(doc['signatory'])

    return page.finish(output_filename)
