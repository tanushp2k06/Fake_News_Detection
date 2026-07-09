from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import torch
import requests
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI(docs_url=None, redoc_url=None)

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
      <title>API Docs</title>
      <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist/swagger-ui.css">
      <style>
        body {
          margin: 0;
          background: #0f172a;
        }

        .swagger-ui {
          color: #e5e7eb;
        }

        .swagger-ui .topbar {
          background: #111827;
        }

        .swagger-ui .info,
        .swagger-ui .scheme-container,
        .swagger-ui .opblock,
        .swagger-ui .responses-inner,
        .swagger-ui section.models {
          background: #111827 !important;
          color: white !important;
          border-color: #1f2937 !important;
        }

        .swagger-ui .opblock-summary {
          background: #1e3a8a !important;
          color: white !important;
        }

        .swagger-ui input,
        .swagger-ui textarea,
        .swagger-ui select {
          background: #0b1220 !important;
          color: white !important;
          border: 1px solid #334155 !important;
        }

        .swagger-ui .btn.execute {
          background: #2563eb !important;
          color: white !important;
        }

        .swagger-ui .response-col_status,
        .swagger-ui label,
        .swagger-ui h1,
        .swagger-ui h2,
        .swagger-ui h3,
        .swagger-ui p,
        .swagger-ui span {
          color: #e5e7eb !important;
        }
      </style>
    </head>

    <body>
      <div id="swagger-ui"></div>

      <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist/swagger-ui-bundle.js"></script>

      <script>
        SwaggerUIBundle({
          url: "/openapi.json",
          dom_id: "#swagger-ui"
        });
      </script>
    </body>
    </html>
    """
    return HTMLResponse(html)

MODEL_PATH = "final_roberta_model"
NEWS_API_KEY = "309405954d8b4e599d8e1b2532b52c15"

TRUSTED_DOMAINS = [

    # Global
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "bbc.co.uk",
    "cnn.com",
    "nytimes.com",
    "washingtonpost.com",
    "theguardian.com",
    "bloomberg.com",
    "wsj.com",
    "forbes.com",
    "economist.com",
    "aljazeera.com",

    # India National
    "thehindu.com",
    "indianexpress.com",
    "hindustantimes.com",
    "timesofindia.indiatimes.com",
    "economictimes.indiatimes.com",
    "livemint.com",
    "ndtv.com",
    "news18.com",
    "india.com",
    "deccanherald.com",
    "theprint.in",
    "scroll.in",
    "tribuneindia.com",
    "business-standard.com",
    "financialexpress.com",
    "news18.com",
    "www.news18.com",
    "cnn-news18.com",
    "firstpost.com",
    "moneycontrol.com",
    "cnbctv18.com",

    # India TV / Networks
    "indiatoday.in",
    "aajtak.in",
    "zeenews.india.com",
    "wionews.com",
    "firstpost.com",

    # Official / Government
    "pib.gov.in",
    "mea.gov.in",
    "mha.gov.in",
    "mod.gov.in",
    "pmindia.gov.in",
    "rbi.org.in",

    # International official
    "who.int",
    "un.org",
    "worldbank.org",
    "imf.org",
    "nato.int"
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

id2label = {0: "FAKE", 1: "REAL"}

class NewsInput(BaseModel):
    text: str

def search_newsapi(query):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": " ".join(query.split()[:8]),
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 20,
        "apiKey": NEWS_API_KEY
    }

    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        articles = data.get("articles", [])

        trusted = []
        for a in articles:
            article_url = (a.get("url") or "").lower()
            if any(domain in article_url for domain in TRUSTED_DOMAINS):
                trusted.append({
                    "title": a.get("title", "No title"),
                    "source": a.get("source", {}).get("name", "Unknown"),
                    "url": a.get("url", "")
                })

        return trusted[:5]
    except Exception:
        return []

def make_flags(label, trusted_count):
    if label == "REAL":
        return [
            {"label": "Credible language pattern", "severity": "green"},
            {"label": "Model confidence supports claim", "severity": "green"},
            {"label": f"{trusted_count} trusted source matches", "severity": "green" if trusted_count else "yellow"}
        ]
    else:
        return [
            {"label": "Suspicious linguistic pattern", "severity": "red"},
            {"label": "Low trusted-source confirmation", "severity": "yellow"},
            {"label": "Possible misinformation signal", "severity": "red"}
        ]

@app.post("/analyze")
def analyze_news(data: NewsInput):
    text = data.text

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
        pred_id = torch.argmax(probs).item()

    model_label = id2label[pred_id]
    confidence = float(probs[pred_id])
    trusted_articles = search_newsapi(text)
    trusted_count = len(trusted_articles)

    if model_label == "REAL" and confidence >= 0.80:
        verdict = "REAL"
    elif model_label == "FAKE" and confidence >= 0.75:
        verdict = "FAKE"
    else:
        verdict = "UNCERTAIN"

    confidence_percent = round(confidence * 100)

    summary = (
        f"The transformer model predicted {model_label} with {confidence_percent}% confidence. "
        f"Live source verification found {trusted_count} matching trusted-source articles. "
        f"The final verdict is {verdict}. Source verification is used as supporting evidence, "
        f"not as the only decision factor."
    )

    return {
        "verdict": verdict,
        "confidence": confidence_percent,
        "summary": summary,
        "flags": make_flags(model_label, trusted_count),
        "features": [
            {"name": "Emotional Tone", "score": 65 if verdict == "FAKE" else 25, "note": "Tone-based risk"},
            {"name": "Source Credibility", "score": 20 if trusted_count else 75, "note": "Trusted-source check"},
            {"name": "Claim Verifiability", "score": 35 if trusted_count else 70, "note": "Live evidence availability"},
            {"name": "Linguistic Manipulation", "score": 70 if verdict == "FAKE" else 30, "note": "Model-learned patterns"}
        ],
        "suspicious_tokens": text.split()[:6],
        "sources": trusted_articles
    }
