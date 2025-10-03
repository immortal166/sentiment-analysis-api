from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from textblob import TextBlob

app = FastAPI(
    title="Sentiment Analysis API",
    description="Real-time sentiment analysis using machine learning",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)
class AnalysisRequest(BaseModel):
    text: str

class AnalysisResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    language: str = "en"

def analyze_sentiment_simple(text: str):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    
    if polarity > 0.1:
        return "positive", 0.7 + (polarity * 0.3)
    elif polarity < -0.1:
        return "negative", 0.7 + (abs(polarity) * 0.3)
    else:
        return "neutral", 0.6

@app.post("/analyze", response_model=AnalysisResponse)
def analyze_sentiment(request: AnalysisRequest):
    """
    Analyze sentiment of input text
    Returns: sentiment (positive/negative/neutral) and confidence score
    """
    sentiment, confidence = analyze_sentiment_simple(request.text)
    
    return AnalysisResponse(
        text=request.text,
        sentiment=sentiment,
        confidence=round(confidence, 4),
        language="en"
    )

@app.get("/")
def root():
    return {"status": "active", "service": "Sentiment Analysis API"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}