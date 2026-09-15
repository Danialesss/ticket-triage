"""Stage 3 (serve): a tiny FastAPI endpoint around the trained model.

Run:
    uvicorn app:app --reload
Then POST to /predict with {"text": "..."}.

This is the seam where the roadmap picks up: deploy this to Azure or
DigitalOcean, add CI/CD, and point Datadog at it.
"""
from fastapi import FastAPI
from pydantic import BaseModel

from src.predict import load_artifacts, predict

app = FastAPI(title="Ticket triage")
model, vocab, id2label = load_artifacts("artifacts")


class Ticket(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def classify(ticket: Ticket):
    label, confidence = predict(ticket.text, model, vocab, id2label)
    return {"category": label, "confidence": round(confidence, 4)}
