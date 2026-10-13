# Data Quality Report (Sprint 1)

Window: 2024-01-01 to 2025-12-31

## Row counts
- Rows read from the archive: 16,558,052
- Complaints inside the window: 8,177,541
- Of those, complaints with a written story: 2,032,564
- Final text sample after cleaning and capping: 45,597

## All complaints in the window, by product

| Product | Complaints | Share |
|---|---|---|
| Credit reporting | 7,176,003 | 87.8% |
| Debt collection | 439,178 | 5.4% |
| Credit card | 165,991 | 2.0% |
| Bank account | 137,028 | 1.7% |
| Money transfer | 101,376 | 1.2% |
| Mortgage | 46,153 | 0.6% |
| Student loan | 35,970 | 0.4% |
| Vehicle loan | 34,139 | 0.4% |
| Personal loan | 22,186 | 0.3% |
| Prepaid card | 12,829 | 0.2% |
| Debt management | 6,688 | 0.1% |

## Text sample, by product (capped at 5,000 each)

| Product | Complaints in sample |
|---|---|
| Credit reporting | 5,000 |
| Bank account | 5,000 |
| Mortgage | 5,000 |
| Money transfer | 5,000 |
| Credit card | 5,000 |
| Debt collection | 5,000 |
| Vehicle loan | 4,766 |
| Student loan | 4,667 |
| Personal loan | 3,294 |
| Prepaid card | 1,840 |
| Debt management | 1,030 |

## Cleaning rules
1. Kept complaints received inside the window above.
2. Grouped product names by keyword, since the CFPB has renamed them over the years.
3. Removed the CFPB's privacy redactions (XXXX) and dollar placeholders from the text.
4. Dropped stories under 50 characters and exact duplicate stories.
5. Capped each product at 5,000 complaints so credit reporting doesn't swamp the rest.