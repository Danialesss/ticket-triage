# Ticket triage

Small ML project that sorts support tickets into categories. I built it to get hands on with PyTorch and PySpark in one pipeline, and to have something that runs end to end instead of a notebook that stops at "model trained".

Data prep runs in PySpark, the classifier is a small PyTorch model, and a FastAPI endpoint lets you send it a ticket and get a category back over HTTP.

## How it works

Three stages, each its own file:

1. `src/prep_spark.py` reads the ticket CSV in Spark, cleans the text, drops duplicates, adds a couple of features (word count, char count), splits it 80/20, and writes Parquet.
2. `src/train.py` loads that Parquet, builds a vocabulary, and trains the model in `src/model.py`. The model turns each word into a vector, averages them across the ticket, and runs that through a small two layer network. It saves the weights, vocab, and label maps to `artifacts/`.
3. `src/predict.py` and `app.py` load those artifacts and classify new tickets, one from the command line, the other over an API.

## Two datasets

I started with a 40 row sample I wrote by hand, just to get the pipeline working end to end. Once that worked I swapped in a real dataset, about 12,000 English support tickets pulled from a public Hugging Face dataset (`Tobi-Bueck/customer-support-tickets`), sorted into 10 categories like Technical Support, Billing and Payments, Human Resources, and so on.

`src/fetch_data.py` downloads that dataset, keeps the English rows, and combines the subject and body into one text column so it lines up with the format the rest of the pipeline expects.

## Running it

You need Python and a Java runtime (Spark needs Java). Codespaces already has both, which is where I built this.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

```bash
# 0. get the real dataset (or skip this and use data/sample_tickets.csv)
python -m src.fetch_data

# 1. data prep
export PYSPARK_PYTHON=$(which python)
python -m src.prep_spark --input data/support_tickets_en.csv --output data/processed_full

# 2. train
python -m src.train --train data/processed_full/train --test data/processed_full/test --epochs 40

# 3a. predict from the CLI
python -m src.predict "I was charged twice this month"

# 3b. or serve it
uvicorn app:app --reload
# open the forwarded URL, add /docs, try the /predict endpoint
```

Tests (pure Python, no Spark or Torch needed, this is what CI runs):

```bash
pytest tests/ -v
```

## What I found on the 40 row sample

This was the first thing I ran, before I had real data.

At 15 epochs the model basically guessed the biggest category for everything, about 38% accuracy. I pushed it to 80 epochs expecting that to help. Training loss fell off a cliff, 1.38 down to 0.14, but test accuracy actually got worse, not better.

That's overfitting. With only 32 training rows the model memorised them instead of learning anything general, and more epochs just made the memorising worse. The 8 row test set also meant the numbers were mostly noise, one random split even left an entire category out of the test set.

## What I found on the real dataset

Once I had close to 12,000 real tickets across 10 categories, the picture changed completely.

At 20 epochs: 42% accuracy. At 40 epochs: 44%, and every category held steady or improved, including a category that had scored a flat 0 at 20 epochs. I kept going to 65 epochs to see if there was more to get. Training loss kept dropping the whole way, down to 0.30, but accuracy stayed at 44%. Some categories gained a little, a couple gave a little back. That's the plateau, the model had learned what it could from this much data and this architecture, and further training was just starting to memorise rather than generalise.

For reference, always guessing the single biggest category on this dataset would get you about 29%. So 44% is a real result, not the model doing nothing.

The category sizes explain most of what the model gets right or wrong. Billing and Payments had over a thousand training examples and scored 0.73 f1, the best of the ten. General Inquiry had only 131 examples and barely scored above 0, the model never really learned what it looks like. Technical Support had the most examples by far but only landed around 0.50, probably because its wording overlaps with IT Support and Product Support, so the model confuses similar sounding problems with each other.

## What I learned

- Falling training loss on its own tells you nothing. What counts is how it does on data it hasn't seen.
- More training doesn't fix too little data, on the 40 row set it made things worse.
- More training on a bigger dataset does help, up to a point, then it plateaus and starts trading generalisation for memorisation, same failure mode, just later.
- A small test set makes the metrics mostly noise. A few thousand rows makes them mean something.
- Category imbalance shows up directly in the per-category scores. The model is only as good as the data behind each category.

## Next

The clear next step isn't more epochs, it's fixing the imbalance, either by dropping the tiniest categories or weighting the loss so rare ones count for more during training. After that I'd look at a stronger model, something like a small fine tuned transformer instead of the averaging classifier, and add more features in the Spark stage.

## Stack

PySpark, PyTorch, pandas, scikit-learn (metrics only), FastAPI, pytest, GitHub Actions.
