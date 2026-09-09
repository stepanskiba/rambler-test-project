from typing import Annotated, Literal

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, StringConstraints

model = joblib.load("models/model.joblib")
app = FastAPI()

ALLOW_BELOW, BLOCK_ABOVE = 0.3, 0.8

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=5000)]

class ModerateRequest(BaseModel):
    text: Text

class ModerateResponse(BaseModel):
    decision: Literal["ALLOW", "REVIEW", "BLOCK"]
    confidence: float

@app.get("/health")
def health() -> bool:
    return True

@app.post("/moderate", response_model=ModerateResponse)
def moderate(req: ModerateRequest) -> ModerateResponse:
    p_toxic = float(model.predict_proba([req.text])[0, 1])

    if p_toxic >= BLOCK_ABOVE:
        decision = "BLOCK"
    elif p_toxic < ALLOW_BELOW:
        decision = "ALLOW"
    else:
        decision = "REVIEW"

    return ModerateResponse(decision=decision, confidence=round(p_toxic, 4))
