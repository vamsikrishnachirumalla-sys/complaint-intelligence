# Sprint 1 - get the complaint data in shape
#
# the archive is ~1GB of parquet, so I stream it in batches and keep only:
#   1. monthly complaint counts for everything (for the trend charts)
#   2. a capped sample of complaints that have a written story (for the text work)
#
#   python src/sprint1_data.py

import re

import pandas as pd
import pyarrow.parquet as pq

from config import (ARCHIVE_DIR, END_DATE, HF_ARCHIVE, MAX_PER_PRODUCT, MONTHLY_FILE,
                    RANDOM_STATE, REPORTS_DIR, SAMPLE_FILE, START_DATE)

KEEP = ["Complaint ID", "Date received", "Product", "Issue", "Company", "State",
        "Consumer complaint narrative", "Company response to consumer", "Timely response?"]


def snake(name):
    # "Timely response?" and "timely_response" both become timely_response,
    # so the same code works on the official csv names and the archive's names
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def download():
    # first try was the official CFPB export, but it came back with no narrative
    # column at all. turns out they stopped publishing the stories in aug 2026,
    # so this pulls an archived copy from before that
    files = sorted(ARCHIVE_DIR.rglob("*.parquet"))
    if files:
        print(f"using the {len(files)} archive files already on disk")
        return files

    print("downloading the complaint archive (~1GB, a few minutes)...")
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=HF_ARCHIVE, repo_type="dataset",
                      allow_patterns=["data/*.parquet"], local_dir=ARCHIVE_DIR)
    return sorted(ARCHIVE_DIR.rglob("*.parquet"))


def match_columns(path):
    header = pq.ParquetFile(path).schema_arrow.names
    lookup = {snake(h): h for h in header}

    found, missing = {}, []
    for want in KEEP:
        real = lookup.get(snake(want))
        if real is None:
            missing.append(want)
        else:
            found[real] = want

    if missing:
        raise SystemExit(f"couldn't find these columns: {missing}\n"
                         f"the file has: {header}")
    return found


def short_product(name):
    # cfpb renames products every few years, keywords survive that.
    # prepaid has to come before credit card or it gets swallowed
    n = str(name).lower()
    if "credit report" in n:
        return "Credit reporting"
    if "debt collection" in n:
        return "Debt collection"
    if "prepaid" in n:
        return "Prepaid card"
    if "credit card" in n:
        return "Credit card"
    if "checking" in n or "savings" in n or "bank account" in n:
        return "Bank account"
    if "mortgage" in n:
        return "Mortgage"
    if "money transfer" in n or "virtual currency" in n:
        return "Money transfer"
    if "vehicle" in n:
        return "Vehicle loan"
    if "student loan" in n:
        return "Student loan"
    if "payday" in n or "personal loan" in n:
        return "Personal loan"
    if "debt or credit management" in n or "credit repair" in n:
        return "Debt management"
    return "Other"


def clean_text(text):
    # the CFPB blacks out names, dates and account numbers as XXXX.
    # those carry no meaning and would end up as "top words" if I left them in
    text = re.sub(r"[Xx]{2,}(/[Xx]{2,})*", " ", str(text))
    text = re.sub(r"\{\$[\d,.]+\}", " ", text)  # dollar amounts look like {$500.00}
    return re.sub(r"\s+", " ", text).strip()


def stream(files):
    monthly_parts, sample_parts = [], []
    stats = {"rows_read": 0, "in_window": 0, "with_story": 0}
    columns = match_columns(files[0])

    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=200_000, columns=list(columns)):
            chunk = batch.to_pandas().rename(columns=columns)
            stats["rows_read"] += len(chunk)

            # some files store the date as a real date, some as text. same format either way
            chunk["Date received"] = pd.to_datetime(chunk["Date received"], errors="coerce").dt.strftime("%Y-%m-%d")
            chunk = chunk.fillna("")
            chunk = chunk[(chunk["Date received"] >= START_DATE) & (chunk["Date received"] <= END_DATE)]
            stats["in_window"] += len(chunk)
            if chunk.empty:
                continue

            chunk = chunk.assign(product=chunk["Product"].map(short_product),
                                 month=chunk["Date received"].str[:7])

            # counts for every complaint, story or not
            monthly_parts.append(chunk.groupby(["month", "product", "Issue"]).size().rename("complaints"))

            stories = chunk[chunk["Consumer complaint narrative"].astype(str).str.len() > 50]
            stats["with_story"] += len(stories)
            # a slice from every batch so the sample spreads across all the files.
            # the per-product cap happens after the loop
            sample_parts.append(stories.sample(frac=0.25, random_state=RANDOM_STATE))

    if not monthly_parts:
        raise SystemExit(f"no complaints between {START_DATE} and {END_DATE}, check the dates in config.py")

    monthly = (pd.concat(monthly_parts).groupby(level=[0, 1, 2]).sum()
                 .reset_index().rename(columns={"Issue": "issue"}))
    sample = pd.concat(sample_parts, ignore_index=True)
    return monthly, sample, stats


def finish_sample(sample):
    sample = sample.rename(columns={
        "Complaint ID": "complaint_id", "Date received": "date", "Issue": "issue",
        "Company": "company", "State": "state", "Consumer complaint narrative": "narrative",
        "Company response to consumer": "company_response", "Timely response?": "timely",
    }).drop(columns=["Product"])

    sample["narrative"] = sample["narrative"].map(clean_text)
    sample = sample[sample["narrative"].str.len() > 50]
    sample = sample.drop_duplicates("narrative")  # some people paste the same story twice
    sample = sample[sample["product"] != "Other"]

    # cap each product so the big ones don't drown out the rest
    # shuffle first, then take the first N per product = a random N per product
    sample = (sample.sample(frac=1, random_state=RANDOM_STATE)
                    .groupby("product").head(MAX_PER_PRODUCT))
    sample["date"] = pd.to_datetime(sample["date"])
    return sample.reset_index(drop=True)


def write_report(stats, monthly, sample):
    by_product = sample["product"].value_counts()
    all_by_product = monthly.groupby("product")["complaints"].sum().sort_values(ascending=False)

    out = [
        "# Data Quality Report (Sprint 1)",
        "",
        f"Window: {START_DATE} to {END_DATE}",
        "",
        "## Row counts",
        f"- Rows read from the archive: {stats['rows_read']:,}",
        f"- Complaints inside the window: {stats['in_window']:,}",
        f"- Of those, complaints with a written story: {stats['with_story']:,}",
        f"- Final text sample after cleaning and capping: {len(sample):,}",
        "",
        "## All complaints in the window, by product",
        "",
        "| Product | Complaints | Share |",
        "|---|---|---|",
    ]
    total = all_by_product.sum()
    for product, n in all_by_product.items():
        out.append(f"| {product} | {n:,} | {n / total:.1%} |")

    out += [
        "",
        f"## Text sample, by product (capped at {MAX_PER_PRODUCT:,} each)",
        "",
        "| Product | Complaints in sample |",
        "|---|---|",
    ]
    for product, n in by_product.items():
        out.append(f"| {product} | {n:,} |")

    out += [
        "",
        "## Cleaning rules",
        "1. Kept complaints received inside the window above.",
        "2. Grouped product names by keyword, since the CFPB has renamed them over the years.",
        "3. Removed the CFPB's privacy redactions (XXXX) and dollar placeholders from the text.",
        "4. Dropped stories under 50 characters and exact duplicate stories.",
        f"5. Capped each product at {MAX_PER_PRODUCT:,} complaints so credit reporting doesn't swamp the rest.",
    ]
    (REPORTS_DIR / "data_quality_report.md").write_text("\n".join(out))


if __name__ == "__main__":
    files = download()
    monthly, sample, stats = stream(files)
    sample = finish_sample(sample)

    monthly.to_csv(MONTHLY_FILE, index=False)
    sample.to_parquet(SAMPLE_FILE, index=False)
    write_report(stats, monthly, sample)

    print(f"rows read:          {stats['rows_read']:,}")
    print(f"in the window:      {stats['in_window']:,}")
    print(f"with a story:       {stats['with_story']:,}")
    print(f"final text sample:  {len(sample):,}")
    print(sample["product"].value_counts().to_string())
