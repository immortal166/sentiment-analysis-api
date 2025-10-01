import requests
import time

def test_sentiment_api():
    print("Testing Sentiment Analysis API")
    print("=" * 40)
    
    test_cases = [
        "I absolutely love this product! It's amazing!",
        "This is the worst experience ever.",
        "The weather is nice today.",
        "The service was average, nothing special.",
        "This product exceeded all my expectations!"
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"\nTest {i}: {text}")
        
        try:
            start_time = time.time()
            response = requests.post(
                "http://localhost:8000/analyze",
                json={"text": text},
                timeout=10
            )
            response_time = round((time.time() - start_time) * 1000, 2)
            
            if response.status_code == 200:
                data = response.json()
                print(f"   Sentiment: {data['sentiment']}")
                print(f"   Confidence: {data['confidence']}")
                print(f"   Response time: {response_time}ms")
            else:
                print(f"   Error: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"   Failed: {e}")

if __name__ == "__main__":
    test_sentiment_api()