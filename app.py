from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import torch

app = FastAPI(
    title="Sentiment Analysis API",
    description="Real-time sentiment analysis using machine learning",
    version="1.0"
)

# Initializing sentiment analysis model
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1
)

class AnalysisRequest(BaseModel):
    text: str

class AnalysisResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    language: str = "en"

@app.post("/analyze", response_model=AnalysisResponse)
def analyze_sentiment(request: AnalysisRequest):
    """
    Analyze sentiment of input text
    Returns: sentiment (positive/negative/neutral) and confidence score
    """
    result = sentiment_analyzer(request.text)[0]
    
    return AnalysisResponse(
        text=request.text,
        sentiment=result['label'].lower(),
        confidence=round(result['score'], 4),
        language="en"
    )

@app.get("/")
def root():
    return {"status": "active", "service": "Sentiment Analysis API"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}