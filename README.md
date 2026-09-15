# Ticket triage

A small, standalone ML repo: classify support tickets into `billing`,
`technical`, `account`, or `general`. Distributed data prep in PySpark,
a compact text classifier in PyTorch, and an optional FastAPI endpoint.

It runs on CPU against the tiny sample dataset included in `data/`, so you can
train and predict end to end in a couple of minutes.

## Pipeline

```
PySpark  ->  PyTorch  ->  serve
clean +      train        FastAPI /
features     classifier   CLI
```

- Stage 1 `src/prep_spark.py` cleans and dedupes tickets, adds simple features
  (`word_count`, `char_count`), splits train/test, writes Parquet.
- Stage 2 `src/train.py` builds a vocab, trains the classifier, prints a
  classification report, and saves artifacts.
- Stage 3 `src/predict.py` (CLI) and `app.py` (FastAPI) score new tickets.

The Spark step is overkill on 40 rows on purpose. The same script scales to
millions of rows on Databricks Community or a cluster, which is the pattern
worth showing.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

PySpark needs a Java runtime (JDK 8, 11, or 17). If you don't have Java yet,
you can run the Spark stage for free on Databricks Community Edition, or skip
straight to training on the raw CSV (see below).

## Run it

```bash
# stage 1: data prep (writes data/processed/train and /test)
python -m src.prep_spark --input data/sample_tickets.csv --output data/processed

# stage 2: train (saves model.pt, vocab.json, labels.json to artifacts/)
python -m src.train --train data/processed/train --test data/processed/test

# stage 3a: predict from the CLI
python -m src.predict "I was charged twice this month"

# stage 3b: serve
uvicorn app:app --reload
# then POST {"text": "..."} to http://127.0.0.1:8000/predict
```

No Java handy? Smoke-test the model on the raw CSV instead:

```bash
python -m src.train --train data/sample_tickets.csv --test data/sample_tickets.csv
```

## Tests

Pure-Python unit tests run in CI without Spark or Torch:

```bash
pip install pytest && pytest tests/ -v
```

## Rough timeline

- PySpark prep: 3 to 4 days
- PyTorch classifier: 4 to 5 days
- Serving + CLI polish: 3 to 4 days

## Next steps

- Swap the mean-pool classifier for an LSTM or a fine-tuned DistilBERT.
- Point it at a real dataset (public support-ticket corpora work well).
- Add more features in the Spark stage (sender domain, time of day, keywords).
