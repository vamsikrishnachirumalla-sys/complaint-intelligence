# Agile Backlog: Customer Complaint Intelligence

## Product vision

Help a bank's customer experience and compliance teams see what customers are complaining about, catch rising problems early, find similar past cases fast, and route new complaints to the right team without someone reading every one.

## Roles

| Role | Person | Cares about |
|---|---|---|
| Product owner | Head of Customer Experience | Spotting problems before they grow |
| End users | Complaint handling agents | Getting each complaint to the right team, seeing how similar cases were resolved |
| Stakeholder | Compliance | Regulator-visible complaint trends |
| Data analyst | Vamsi Krishna Chirumalla | Building and explaining the analysis |

## Definition of done

The script runs from a clean checkout, outputs land in `reports/`, results are checked against a simple baseline, and findings are written in plain language for the person who asked.

---

## Epic 1: Data pipeline (Sprint 1)

**US-1.1** As an analyst, I need complaints from a stable time window with consistent product names, so trends compare like with like.
- Acceptance: product names grouped by keyword; window set in one config file.

**US-1.2** As an analyst, I need a clean, balanced text sample so no single product dominates the models.
- Acceptance: redactions removed, duplicates dropped, each product capped; counts reported before and after.

## Epic 2: Text and social analytics (Sprint 2)

**US-2.1** As the Head of CX, I want to know what complaints are about without reading them.
- Acceptance: topic model with readable labels; share of complaints per topic.

**US-2.2** As Compliance, I want to see which issues are growing fastest.
- Acceptance: last 3 months vs prior 6, with a minimum volume so tiny issues don't top the list.

**US-2.3** As the Head of CX, I want to know which products make customers angriest.
- Acceptance: sentiment per complaint and per product, with the caveat that tone isn't validity.

## Epic 3: Search (Sprint 3)

**US-3.1** As an agent, I want to type a short description and see the most similar past complaints.
- Acceptance: TF-IDF, BM25, and embedding search compared on the same queries with precision@10 and MRR.

## Epic 4: Neural network routing (Sprint 4)

**US-4.1** As an agent, I want each new complaint routed to the right product team automatically.
- Acceptance: neural network built from scratch, gradient-checked, compared against a logistic regression baseline on a held-out test set.

## Epic 5: Deep learning (Sprint 5)

**US-5.1** As the product owner, I want to know if a transformer is worth the extra cost over the simpler models.
- Acceptance: DistilBERT fine-tuned and tested on the same split as Sprint 4; accuracy, macro F1, and training time reported.

## Epic 6: Dashboard (Sprint 6)

**US-6.1** As all users, I want one place to see trends, topics, search, and routing.
- Acceptance: live Streamlit app that runs on the free tier without downloading the raw data.

---

## Sprint plan

| Sprint | Goal | Course it draws on |
|---|---|---|
| 1 | Data pipeline | Intro to Data Science |
| 2 | Topics, sentiment, rising issues | NLP; Social Media and Web Analytics |
| 3 | Search engine | Information Retrieval |
| 4 | Neural network from scratch | Intro to Neural Networks |
| 5 | DistilBERT fine-tuning | Deep Learning |
| 6 | Dashboard | |

---

## Sprint retrospectives

### Sprint 1
- **What went well:**
- **What didn't:** The official CFPB export came back with no narrative column. The CFPB stopped publishing complaint stories in August 2026.
- **What I'll change:** Switched to an archived copy from before the change, and made column matching flexible so a rename can't break the pipeline again.

### Sprint 2
- **What went well:**
- **What didn't:** The first topic model run had "told", "called", and "said" in every topic.
- **What I'll change:** Added a short list of complaint filler words to the stop words.

### Sprint 3
- **What went well:** Writing BM25 on a sparse matrix made each query a single matrix multiply, fast enough for live search.
- **What didn't:**
- **What I'll change:**

### Sprint 4
- **What went well:** Gradient check matched the hand-written backprop to within one part in a million.
- **What didn't:**
- **What I'll change:**

### Sprint 5
- **What went well:** Once it had the full training set, DistilBERT was the most accurate model (80.8%).
- **What didn't:** The first run used 20,000 complaints and 2 epochs, and DistilBERT lost to the logistic regression baseline (78.5% vs 79.6%).
- **What I'll change:** Retrained on all 31,917 training complaints for 3 epochs before drawing conclusions. Lesson: make sure the fancy model got a fair shot before declaring it a winner or a loser.

### Sprint 6
- **What went well:**
- **What didn't:**
- **What I'll change:**
