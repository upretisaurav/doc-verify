"""Bank statements: generator, consistency rules, templates, renderer and OCR values.

All money is stored as whole cents so the balance arithmetic is exact.
"""

import datetime
import random

from sad_syn.generate import COUNTRIES, FIXED_REFERENCE_DATE, fake
from sad_syn.layout import Page, format_cents, format_date

CREDITS = {
    "Salary payment": (2_000_00, 6_000_00),
    "Transfer from savings": (500_00, 5_000_00),
    "Scholarship payment": (1_000_00, 4_000_00),
    "Family transfer": (500_00, 5_000_00),
    "Refund": (10_00, 300_00),
    "Interest earned": (5_00, 150_00),
}
DEBITS = {
    "Rent": (900_00, 2_200_00),
    "Insurance premium": (60_00, 400_00),
    "Electricity bill": (40_00, 300_00),
    "Groceries": (20_00, 220_00),
    "ATM withdrawal": (20_00, 400_00),
    "Internet bill": (40_00, 110_00),
    "Mobile plan": (15_00, 90_00),
    "Gym membership": (20_00, 90_00),
    "Bookshop": (10_00, 180_00),
    "Restaurant": (12_00, 160_00),
    "Public transport": (5_00, 80_00),
    "Pharmacy": (5_00, 90_00),
}

STATEMENT_PERIOD_DAYS = 30
TRANSACTION_DATE_FORMAT = "%d %b"

BANK_PRINTED_TWICE = ()

BANK_EXPECTED_BROKEN_RULE = {
    "altered_date": "date_order",
    "altered_amount": "closing_balances_match",
    "institution_mismatch": "bank_names_match",
    "total_mismatch": "balance_reconciles",
}


def make_bank_name(country):
    local_fake = COUNTRIES[country]['faker']
    return random.choice([
        f"{local_fake.last_name()} Bank",
        f"Bank of {local_fake.city()}",
        f"{local_fake.city()} Savings Bank",
    ])


def draw_opening_balance():
    # always above the largest possible total of debits, so a closing balance can never go negative
    return random.randint(40_000_00, 90_000_00)


def draw_transactions(period_start):
    transactions = []
    for _ in range(random.randint(8, 14)):
        is_credit = random.random() < 0.3
        description = random.choice(list(CREDITS if is_credit else DEBITS))
        amount = random.randint(*(CREDITS if is_credit else DEBITS)[description])
        transactions.append({
            'date': period_start + datetime.timedelta(days=random.randint(0, STATEMENT_PERIOD_DAYS)),
            'description': description,
            'credit': amount if is_credit else 0,
            'debit': 0 if is_credit else amount,
        })
    return sorted(transactions, key=lambda transaction: transaction['date'])


def make_clean_statement():
    doc = {'doc_type': 'bank_statement'}

    country = random.choice(list(COUNTRIES))
    local_fake = COUNTRIES[country]['faker']

    start_date_obj = FIXED_REFERENCE_DATE - datetime.timedelta(days=365)
    issue_date = fake.date_between(start_date=start_date_obj, end_date=FIXED_REFERENCE_DATE)
    period_end = issue_date - datetime.timedelta(days=random.randint(1, 10))
    period_start = period_end - datetime.timedelta(days=STATEMENT_PERIOD_DAYS)

    opening_balance = draw_opening_balance()
    transactions = draw_transactions(period_start)
    total_credits = sum(transaction['credit'] for transaction in transactions)
    total_debits = sum(transaction['debit'] for transaction in transactions)
    closing_balance = opening_balance + total_credits - total_debits

    bank_name = make_bank_name(country)
    if country == "Australia":
        branch_code = fake.bothify(text='BSB ###-###')
    else:
        branch_code = fake.bothify(text='Sort code ##-##-##')

    doc['bank_header'] = bank_name
    doc['bank_body'] = bank_name
    doc['country'] = country
    doc['currency'] = COUNTRIES[country]['currency']
    doc['contact_info'] = local_fake.address().replace('\n', ', ')
    doc['date_of_issue'] = issue_date
    doc['account_holder'] = fake.name()
    doc['holder_address'] = fake.address().replace('\n', ', ')
    doc['account_number'] = str(fake.random_number(digits=8, fix_len=True))
    doc['branch_code'] = branch_code
    doc['period_start'] = period_start
    doc['period_end'] = period_end
    doc['opening_balance'] = opening_balance
    doc['transactions'] = transactions
    doc['total_credits'] = total_credits
    doc['total_debits'] = total_debits
    doc['closing_balance_summary'] = closing_balance
    doc['closing_balance_body'] = closing_balance
    doc['signatory'] = local_fake.name() + ", Branch Manager"

    return doc


def check_statement_rules(doc):
    rules = {}

    rules['date_order'] = doc['period_start'] < doc['period_end'] < doc['date_of_issue']

    rules['closing_balances_match'] = (doc['closing_balance_summary'] == doc['closing_balance_body'])

    rules['bank_names_match'] = (doc['bank_header'] == doc['bank_body'])

    expected_closing = doc['opening_balance'] + doc['total_credits'] - doc['total_debits']
    rules['balance_reconciles'] = (doc['closing_balance_summary'] == expected_closing)

    return rules


def make_inconsistent_statement(kind):
    doc = make_clean_statement()

    if kind == "altered_date":
        doc['altered_field'] = 'period_end'
        doc['original_value'] = doc['period_end']

        gap_days = random.randint(1, 45)

        if random.choice([True, False]):
            doc['period_end'] = doc['period_start'] - datetime.timedelta(days=gap_days)
        else:
            doc['period_end'] = doc['date_of_issue'] + datetime.timedelta(days=gap_days)

    elif kind == "altered_amount":
        # only the body copy is altered: the summary copy must still reconcile with the opening balance
        doc['altered_field'] = 'closing_balance_body'
        doc['original_value'] = doc['closing_balance_body']

        other_opening = doc['opening_balance']
        while other_opening == doc['opening_balance']:
            other_opening = draw_opening_balance()

        doc['closing_balance_body'] = other_opening + doc['total_credits'] - doc['total_debits']

    elif kind == "institution_mismatch":
        doc['altered_field'] = 'bank_body'
        doc['original_value'] = doc['bank_body']

        new_name = doc['bank_body']
        while new_name == doc['bank_body']:
            new_name = make_bank_name(doc['country'])

        doc['bank_body'] = new_name

    elif kind == "total_mismatch":
        doc['altered_field'] = 'opening_balance'
        doc['original_value'] = doc['opening_balance']

        wrong_opening = doc['opening_balance']
        while wrong_opening == doc['opening_balance']:
            wrong_opening = draw_opening_balance()

        doc['opening_balance'] = wrong_opening

    doc['inconsistency_type'] = kind

    return doc


def statement_ocr_fields(doc, template):
    currency = doc['currency']
    return {
        'bank_header': doc['bank_header'],
        'bank_body': doc['bank_body'],
        'opening_balance': format_cents(doc['opening_balance'], currency),
        'total_credits': format_cents(doc['total_credits'], currency),
        'total_debits': format_cents(doc['total_debits'], currency),
        'closing_balance_summary': format_cents(doc['closing_balance_summary'], currency),
        'closing_balance_body': format_cents(doc['closing_balance_body'], currency),
        'period_start': format_date(doc['period_start'], template),
        'period_end': format_date(doc['period_end'], template),
        'date_of_issue': format_date(doc['date_of_issue'], template),
    }


def render_statement(doc, template, output_filename=None):
    page = Page(template)
    fields = statement_ocr_fields(doc, template)

    def draw_summary():
        summary_rows = [
            ("Opening balance:", fields['opening_balance']),
            ("Total credits:", fields['total_credits']),
            ("Total debits:", fields['total_debits']),
            ("Closing balance:", fields['closing_balance_summary']),
        ]
        offsets = page.columns(0, 0.62)
        page.line("Account summary")
        for label, value in summary_rows:
            page.row([label, value], offsets)
        page.blank()

    def draw_transactions():
        offsets = page.columns(0, 0.14, 0.62, 0.81)
        page.line("Transactions")
        page.row(["Date", "Description", "Debit", "Credit"], offsets)
        for transaction in doc['transactions']:
            page.row(
                [
                    transaction['date'].strftime(TRANSACTION_DATE_FORMAT),
                    transaction['description'],
                    format_cents(transaction['debit']) if transaction['debit'] else "",
                    format_cents(transaction['credit']) if transaction['credit'] else "",
                ],
                offsets,
            )
        page.blank()

    page.header(fields['bank_header'], doc['contact_info'])
    page.blank()
    page.line("Account Statement", align=template['title_align'])
    page.blank()

    page.line(doc['account_holder'])
    page.line(doc['holder_address'])
    page.line(f"Account number: {doc['account_number']} | {doc['branch_code']}")
    page.line(f"Statement period: {fields['period_start']} to {fields['period_end']}")
    page.line(f"Date of issue: {fields['date_of_issue']}")
    page.blank()

    if template['summary_first']:
        draw_summary()
        draw_transactions()
    else:
        draw_transactions()
        draw_summary()

    page.paragraph(
        f"{fields['bank_body']} confirms that the closing balance of this account "
        f"was {fields['closing_balance_body']}."
    )
    page.blank()
    page.line(doc['signatory'])

    return page.finish(output_filename)
