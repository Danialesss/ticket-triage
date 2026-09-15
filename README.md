# Ticket triage

Small ML project that sorts support tickets into billing, technical, account, or general. I built it to get hands on with PyTorch and PySpark in one pipeline, and to have something that runs end to end instead of a notebook that stops at "model trained".

Data prep runs in PySpark, the classifier is a small PyTorch model, and a FastAPI endpoint lets you send it a ticket and get a category back over HTTP.

## How it works

Three stages, each its own file:

1. `src/prep_spark.py` reads the raw ticket CSV in Spark, cleans the text, drops duplicates, adds a couple of features (word count, char count), splits it 80/20, and writes Parquet. Spark is overkill for the tiny sample here, but the same code runs unchanged on a real dataset with millions of rows, which was the point of using it.
2. `src/train.py` loads that Parquet, builds a vocabulary, and trains the model in `src/model.py`. The model turns each word into a vector, averages them across the ticket, and runs that through a small two layer network. It saves the weights, vocab, and label maps to `artifacts/`.
3. `src/predict.py` and `app.py` load those artifacts and classify new tickets, one from the command line, the other over an API.

## Running it

You need Python and a Java runtime (Spark needs Java). Codespaces already has both, which is where I built this.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

```bash
# 1. data prep
export PYSPARK_PYTHON=$(which python)
python -m src.prep_spark --input data/sample_tickets.csv --output data/processed

# 2. train
python -m src.train --train data/processed/train --test data/processed/test

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

## What I found

The sample set is deliberately tiny, 40 tickets, so the interesting part was watching how the model behaves on almost no data.

At 15 epochs it basically guessed "general" for everything, about 38% accuracy. My first instinct was to train longer, so I pushed it to 80 epochs. Training loss then fell off a cliff, from 1.38 down to 0.14, but test accuracy actually got worse instead of better.

That gap is textbook overfitting. With only 32 training rows the model memorised them rather than learning anything general, and more epochs just made the memorising worse. The 8 row test set made the numbers bounce around on noise too. One random split even dropped every account ticket into training and left none for testing, so that class couldn't be scored at all.

Oddly, the overfit model still predicts well on tickets worded like the training data. "how do I reset my password" comes back as account at 96%, and the API tags "my invoice is wrong" as billing at 95%. So it is fine for a demo, as long as you know why.

## What I learned

- Falling training loss on its own tells you nothing. What counts is how it does on data it has not seen.
- More training does not fix too little data. It made things worse here.
- On a small test set the metrics are mostly noise, so I stopped reading too much into the exact accuracy.
- Code and data are separate problems. The pipeline is correct, the dataset is the ceiling.

## Next

The real next step is more data. This same pipeline would take a public support ticket dataset of a few thousand rows with no code changes. After that I would swap the averaging model for something stronger like a small fine tuned transformer, and add more features in the Spark stage.

## Stack

PySpark, PyTorch, pandas, scikit-learn (metrics only), FastAPI, pytest, GitHub Actions.
