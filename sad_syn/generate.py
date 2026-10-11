"""Synthetic offer-letter generation: clean offers, single-rule inconsistencies, and rule checks."""

import datetime
import random

from faker import Faker

fake = Faker(['en_AU', 'en_GB'])

COUNTRIES = {
    "Australia": {"faker": Faker('en_AU'), "currency": "AUD"},
    "United Kingdom": {"faker": Faker('en_GB'), "currency": "GBP"},
}

FIXED_REFERENCE_DATE = datetime.date(2026, 10, 1)
VALID_FEES = [n * 500 for n in range(30, 91)]
DEPOSIT_RATES = [0.1, 0.2, 0.25, 0.5]


def set_seed(seed=42):
    Faker.seed(seed)
    random.seed(seed)


def make_university_name(country):
    local_fake = COUNTRIES[country]['faker']
    return random.choice([f"University of {local_fake.city()}", f"{local_fake.last_name()} University"])


def make_clean_offer():
    offer = {'doc_type': 'offer_letter'}

    start_date_obj = FIXED_REFERENCE_DATE - datetime.timedelta(days=365)
    issue_date = fake.date_between(start_date=start_date_obj, end_date=FIXED_REFERENCE_DATE)

    payment_deadline = issue_date + datetime.timedelta(days=random.randint(14, 30))
    orientation_date = payment_deadline + datetime.timedelta(days=random.randint(30, 90))
    commencement_date = orientation_date + datetime.timedelta(days=7)

    degree_level = random.choice(["Bachelor's", "Master's", "PhD"])

    if degree_level == "Bachelor's":
        duration_years = random.choice([3.0, 4.0])
    elif degree_level == "Master's":
        duration_years = random.choice([1.0, 1.5, 2.0])
    else:
        duration_years = random.choice([3.0, 4.0])

    completion_date = commencement_date + datetime.timedelta(days=int(duration_years * 365))

    program_subject = random.choice(["Computer Science", "Data Analytics", "Business Administration", "Engineering", "Design"])

    if degree_level == "Bachelor's":
        program_title = f"Bachelor of {program_subject}"
    elif degree_level == "Master's":
        program_title = f"Master of {program_subject}"
    else:
        program_title = f"Doctor of Philosophy in {program_subject}"

    country = random.choice(list(COUNTRIES))
    local_fake = COUNTRIES[country]['faker']
    univ_name = make_university_name(country)

    first_year_fee = random.choice(VALID_FEES)
    total_tuition = round(first_year_fee * duration_years)
    acceptance_deposit = round(first_year_fee * random.choice(DEPOSIT_RATES))

    offer['university_header'] = univ_name
    offer['university_body'] = univ_name
    offer['country'] = country
    offer['contact_info'] = local_fake.address().replace('\n', ', ')
    offer['date_of_issue'] = issue_date
    offer['student_name'] = fake.name()
    offer['applicant_address'] = fake.address().replace('\n', ', ')
    offer['student_id'] = str(fake.random_number(digits=8, fix_len=True))
    offer['application_number'] = fake.bothify(text='APP-########')
    offer['program_title'] = program_title
    offer['degree_level'] = degree_level
    offer['faculty_department'] = random.choice(["Faculty of Engineering", "Faculty of Science", "Faculty of Business", "Faculty of Arts", "Faculty of Computing"])
    offer['study_mode'] = random.choice(["Full-time", "Part-time"])
    offer['campus_location'] = local_fake.city() + " Campus"
    offer['duration_years'] = duration_years
    offer['orientation_date'] = orientation_date
    offer['commencement_date'] = commencement_date
    offer['completion_date'] = completion_date
    offer['currency'] = COUNTRIES[country]['currency']
    offer['first_year_tuition_fee'] = first_year_fee
    offer['total_estimated_tuition'] = total_tuition
    offer['acceptance_deposit_header'] = acceptance_deposit
    offer['acceptance_deposit_body'] = acceptance_deposit
    offer['payment_deadline'] = payment_deadline
    offer['conditions'] = random.choice(["None", "Subject to final academic transcripts", "Subject to English proficiency test"])
    offer['signatory'] = local_fake.name() + ", Admissions Director"

    return offer


def check_rules(offer):
    rules = {}

    rules['date_order'] = (
        offer['date_of_issue'] < offer['payment_deadline'] <
        offer['orientation_date'] < offer['commencement_date'] <
        offer['completion_date']
    )

    expected_completion = offer['commencement_date'] + datetime.timedelta(days=int(offer['duration_years'] * 365))
    rules['completion_matches_duration'] = (offer['completion_date'] == expected_completion)

    expected_total = round(offer['first_year_tuition_fee'] * offer['duration_years'])
    rules['total_matches_fee'] = (offer['total_estimated_tuition'] == expected_total)

    rules['deposit_below_fee'] = (
        offer['acceptance_deposit_header'] < offer['first_year_tuition_fee'] and
        offer['acceptance_deposit_body'] < offer['first_year_tuition_fee']
    )

    rules['deposits_match'] = (offer['acceptance_deposit_header'] == offer['acceptance_deposit_body'])

    rules['university_names_match'] = (offer['university_header'] == offer['university_body'])

    return rules


def make_inconsistent_offer(kind):
    offer = make_clean_offer()

    if kind == "altered_date":
        offer['altered_field'] = 'payment_deadline'
        offer['original_value'] = offer['payment_deadline']

        gap_days = random.randint(1, 45)

        if random.choice([True, False]):
            offer['payment_deadline'] = offer['date_of_issue'] - datetime.timedelta(days=gap_days)
        else:
            offer['payment_deadline'] = offer['orientation_date'] + datetime.timedelta(days=gap_days)

    elif kind == "altered_amount":
        field_to_alter = random.choice(['acceptance_deposit_header', 'acceptance_deposit_body'])

        offer['altered_field'] = field_to_alter
        offer['original_value'] = offer[field_to_alter]

        fee = offer['first_year_tuition_fee']
        other_rates = [r for r in DEPOSIT_RATES if round(fee * r) != offer[field_to_alter]]
        offer[field_to_alter] = round(fee * random.choice(other_rates))

    elif kind == "institution_mismatch":
        offer['altered_field'] = 'university_body'
        offer['original_value'] = offer['university_body']

        new_name = offer['university_body']
        while new_name == offer['university_body']:
            new_name = make_university_name(offer['country'])

        offer['university_body'] = new_name

    elif kind == "total_mismatch":
        original_total = offer['total_estimated_tuition']
        offer['altered_field'] = 'total_estimated_tuition'
        offer['original_value'] = original_total

        other_fees = [f for f in VALID_FEES if f != offer['first_year_tuition_fee']]
        offer['total_estimated_tuition'] = round(random.choice(other_fees) * offer['duration_years'])

    offer['inconsistency_type'] = kind

    return offer


INCONSISTENCY_KINDS = ["altered_date", "altered_amount", "institution_mismatch", "total_mismatch"]

EXPECTED_BROKEN_RULE = {
    "altered_date": "date_order",
    "altered_amount": "deposits_match",
    "institution_mismatch": "university_names_match",
    "total_mismatch": "total_matches_fee",
}


def make_letters(n):
    return [
        make_clean_offer() if i % 2 == 0
        else make_inconsistent_offer(INCONSISTENCY_KINDS[(i // 2) % len(INCONSISTENCY_KINDS)])
        for i in range(n)
    ]

