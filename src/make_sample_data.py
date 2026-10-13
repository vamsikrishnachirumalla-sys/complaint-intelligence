# builds a fake complaints file in the same format as the real CFPB export,
# so I can test every sprint quickly without the 1GB download.
# the complaints are stitched together from templates. never quote findings from it.
#
#   python src/make_sample_data.py

import re

import numpy as np
import pandas as pd

from config import ARCHIVE_DIR, RANDOM_STATE

# product -> issue -> a few phrases a real customer might write
TEMPLATES = {
    "Credit reporting or other personal consumer reports": {
        "Incorrect information on your report": [
            "there is an account on my credit report that does not belong to me",
            "my report shows a late payment that I never made",
            "the balance listed on my credit report is wrong",
        ],
        "Problem with a company's investigation into an existing problem": [
            "I disputed this item three times and the bureau keeps verifying it without proof",
            "they never responded to my dispute within 30 days",
        ],
        "Improper use of your report": [
            "someone pulled my credit without my permission and I see a hard inquiry",
            "there is an inquiry from a lender I never applied with",
        ],
    },
    "Debt collection": {
        "Attempts to collect debt not owed": [
            "a collector keeps calling about a debt I already paid off",
            "this debt is not mine and they refuse to validate it",
        ],
        "Communication tactics": [
            "they call me at work every day even after I asked them to stop",
            "the collector threatened to sue me and called my family members",
        ],
    },
    "Credit card": {
        "Problem with a purchase shown on your statement": [
            "there is a charge on my card I did not make and the bank denied my dispute",
            "I returned the item but the merchant refund never posted to my card",
        ],
        "Fees or interest": [
            "they charged me a late fee even though I paid before the due date",
            "my interest rate went up without any notice",
        ],
    },
    "Checking or savings account": {
        "Managing an account": [
            "my account was frozen and I cannot access my paycheck",
            "the bank closed my checking account without explanation",
        ],
        "Problem with a lender or other company charging your account": [
            "unauthorized withdrawals were taken from my savings account",
            "I was charged overdraft fees on transactions that cleared the same day",
        ],
    },
    "Mortgage": {
        "Trouble during payment process": [
            "my mortgage servicer misapplied my payment and now says I am late",
            "escrow was miscalculated and my monthly payment doubled",
        ],
        "Struggling to pay mortgage": [
            "I applied for a loan modification and the servicer keeps losing my documents",
            "they started foreclosure while my forbearance request was pending",
        ],
    },
    "Money transfer, virtual currency, or money service": {
        "Fraud or scam": [
            "I was scammed through the app and the company refuses to refund the transfer",
            "someone accessed my account and sent money to a stranger",
        ],
        "Other transaction problem": [
            "my wire transfer has been pending for two weeks with no update",
            "the app locked my account with my funds inside",
        ],
    },
    "Vehicle loan or lease": {
        "Managing the loan or lease": [
            "the lender added charges to my car loan that were never explained",
            "my auto loan payoff amount keeps changing",
        ],
        "Problems at the end of the loan or lease": [
            "they repossessed my car even though I was current on payments",
            "the lease company charged excessive wear and tear fees at turn in",
        ],
    },
    "Student loan": {
        "Dealing with your lender or servicer": [
            "my student loan servicer put my loans in forbearance without asking",
            "my income driven repayment application has been processing for months",
        ],
    },
    "Payday loan, title loan, personal loan, or advance loan": {
        "Charged fees or interest you didn't expect": [
            "the personal loan had hidden fees that were never disclosed",
            "the payday lender keeps rolling over my loan and charging more fees",
        ],
    },
}

FILLER = [
    "I have called customer service many times.",
    "Nobody has been able to help me.",
    "This is causing me a lot of stress.",
    "I have documents to prove this.",
    "Please help me resolve this.",
    "I was on hold for hours.",
    "They keep transferring me to different departments.",
    "I expect this to be fixed and corrected immediately.",
    "Thank you for looking into this.",
    "On XX/XX/XXXX I contacted them again.",
]

COMPANIES = ["EQUIFAX, INC.", "Experian Information Solutions Inc.",
             "TRANSUNION INTERMEDIATE HOLDINGS, INC.", "BANK OF AMERICA, NATIONAL ASSOCIATION",
             "CAPITAL ONE FINANCIAL CORPORATION", "JPMORGAN CHASE & CO.", "WELLS FARGO & COMPANY",
             "Block, Inc.", "PayPal Holdings, Inc.", "Navient Solutions, LLC."]
STATES = ["FL", "TX", "CA", "GA", "NY", "IL", "PA", "NC", "OH", "NJ"]

# real data is extremely lopsided toward credit reporting, keep that here too
PRODUCT_WEIGHTS = [0.62, 0.1, 0.07, 0.06, 0.04, 0.04, 0.03, 0.02, 0.02]


def build(rows=40_000):
    rng = np.random.default_rng(RANDOM_STATE)
    products = list(TEMPLATES)
    dates = pd.date_range("2023-06-01", "2026-03-31", freq="D")

    out = []
    for i in range(rows):
        product = rng.choice(products, p=PRODUCT_WEIGHTS)
        issue = rng.choice(list(TEMPLATES[product]))
        date = dates[rng.integers(len(dates))]

        # make "unauthorized" style issues grow over time so the
        # spike detection in sprint 2 has something real to catch
        if "Fraud" in issue and date.year == 2025 and date.month >= 9 and rng.random() < 0.5:
            issue = "Fraud or scam"

        if rng.random() < 0.55:  # a lot of real complaints don't include a story
            main = rng.choice(TEMPLATES[product][issue])
            extra = " ".join(rng.choice(FILLER, size=rng.integers(1, 5), replace=False))
            narrative = f"{main[0].upper()}{main[1:]}. {extra}"
        else:
            narrative = ""

        out.append({
            "Date received": date.strftime("%Y-%m-%d"),
            "Product": product,
            "Sub-product": "",
            "Issue": issue,
            "Sub-issue": "",
            "Consumer complaint narrative": narrative,
            "Company public response": "",
            "Company": rng.choice(COMPANIES),
            "State": rng.choice(STATES),
            "ZIP code": "",
            "Tags": "",
            "Consumer consent provided?": "Consent provided" if narrative else "",
            "Submitted via": "Web",
            "Date sent to company": date.strftime("%Y-%m-%d"),
            "Company response to consumer": rng.choice(["Closed with explanation", "Closed with monetary relief",
                                                        "Closed with non-monetary relief", "In progress"],
                                                       p=[0.75, 0.05, 0.15, 0.05]),
            "Timely response?": rng.choice(["Yes", "No"], p=[0.97, 0.03]),
            "Consumer disputed?": "N/A",
            "Complaint ID": 9_000_000 + i,
        })
    return pd.DataFrame(out)


if __name__ == "__main__":
    fake = build()
    # same layout as the real archive: snake_case names, split over a few parquet files.
    # first file keeps dates as real dates, second as text, since I've seen both
    fake.columns = [re.sub(r"[^a-z0-9]+", "_", c.lower()).strip("_") for c in fake.columns]
    out = ARCHIVE_DIR / "data"
    out.mkdir(parents=True, exist_ok=True)
    half = len(fake) // 2
    first = fake.iloc[:half].copy()
    first["date_received"] = pd.to_datetime(first["date_received"])
    first.to_parquet(out / "complaints-00000.parquet", index=False)
    fake.iloc[half:].to_parquet(out / "complaints-00001.parquet", index=False)
    print(f"wrote {len(fake):,} fake complaints to {out}")
    print("heads up: practice data only, run sprint 1 on the real download before sharing anything")
