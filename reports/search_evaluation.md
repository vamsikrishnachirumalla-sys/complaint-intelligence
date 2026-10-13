# Search Evaluation (Sprint 3)

Searched 45,597 complaints with 300 test queries. Each query is the first 25 words of a held-out complaint.

**Relevance rule:** a result counts as relevant if it has the same product and issue as the complaint the query came from. There are no human relevance labels for this data, so this is a consistent but strict stand-in.

| Method | Precision@10 | Same-product precision@10 | MRR | ms per query |
|---|---|---|---|---|
| TF-IDF | 25.6% | 54.6% | 0.381 | 10.8 |
| BM25 | 28.7% | 59.6% | 0.427 | 1.1 |
| Embeddings | 29.8% | 61.3% | 0.453 | 13.9 |

## Example: top 3 results for "someone used my card without permission and the bank refused to give my money back"

**TF-IDF**

- *Credit card / Improper use of your report*: this company used my information without my permission....
- *Credit card / Problem with a purchase shown on your statement*: Account opened and used in my name without my permission...
- *Credit card / Getting a credit card*: my info was used to obtain credit without my permission...

**BM25**

- *Bank account / Problem with a lender or other company charging your account*: Subject : Unauthorized Withdrawal From My Personal Account to Pay a Business Credit Card Bank of America Bank : Bank of America Manager I spoke with : Date this...
- *Prepaid card / Problem with a purchase or transfer*: Someone used my account debit card and took over and Bank of America denied my claim. Someone made over 30 transactions using my debit card between /23 to /23, ...
- *Prepaid card / Unexpected or other fees*: My card was used without my permission. It was stolen from the hotel I was at. I reported the charges and cancelled my card. They refused to reverse the charges...

**Embeddings**

- *Prepaid card / Problem with a purchase or transfer*: Someone went to the and took my money off my card while I was at work all day with my card on me. I don't understand how someone can take my money off my card a...
- *Prepaid card / Problem with a purchase or transfer*: Someone stolen my card and withdrew funds Im having a hard time getting my money back from this card I file my despite already...
- *Bank account / Problem with a lender or other company charging your account*: The was money taken from my account and my account was also closed I made several dispute and did not get any help my card was being charged over and over withd...
