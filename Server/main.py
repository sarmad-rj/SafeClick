# backend.py
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import json
import whois
from datetime import datetime
from urllib.parse import urlparse
import re
import numpy as np

# ---------------- FastAPI setup ----------------
app = FastAPI(title="Safe Click Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow extension / localhost
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Load model & features ----------------
model = joblib.load("phish_model_deploy.pkl")  # your trained RandomForest model
with open("feature_list.json", "r") as f:
    feature_list = json.load(f)

# ---------------- Request models ----------------
class URLRequest(BaseModel):
    url: str

class DomainRequest(BaseModel):
    domain: str

# ---------------- Helper functions ----------------
SUSPICIOUS_WORDS = [
    "login", "secure", "verify", "update", "password",
    "bank", "free", "gift", "confirm", "account"
]

def extract_features(url):
    url = url.strip()
    parsed = urlparse(url)
    domain = parsed.netloc.lower() if parsed.netloc else ""
    path = parsed.path.lower() if parsed.path else ""

    domain_tokens = domain.split(".")
    
    found_words = [w for w in SUSPICIOUS_WORDS if w in url.lower()]
    suspicious_count = len(found_words)

    # Avoid divide by zero
    path_parts = path.split("/") if path else [""]

    features = {
        "url_length": len(url),
        "domain_length": len(domain),
        "path_length": len(path),
        "num_dots": url.count("."),
        "num_digits": sum(c.isdigit() for c in url),
        "digit_ratio": sum(c.isdigit() for c in url) / max(len(url),1),
        "special_ratio": sum(c in "-@!?=&%" for c in url) / max(len(url),1),
        "has_https": int(parsed.scheme == "https"),
        "has_ip": int(bool(re.search(r'\b\d{1,3}(\.\d{1,3}){3}\b', domain))),
        "subdomain_count": max(domain.count(".") - 1, 0),
        "avg_domain_token_length": int(np.mean([len(t) for t in domain_tokens if t])) if domain_tokens else 0,
        "suspicious_word_count": suspicious_count,
        "suspicious_density": suspicious_count / max(len(path_parts),1)
    }
    return features, found_words

# ---------------- Routes ----------------
@app.post("/predict")
def predict(req: URLRequest):
    try:
        reasons = []
        
        # Step 1: Extract features
        feats, found_words = extract_features(req.url)
        if not feats:
            return {"error": "Invalid URL for feature extraction"}
        
        # Create a DataFrame with feature names to avoid UserWarning
        X = pd.DataFrame([feats], columns=feature_list)
        
        # Step 2: Get phishing probability from model
        phishing_proba = 0.0
        try:
            proba_array = model.predict_proba(X)
            if proba_array is not None and len(proba_array) > 0 and len(proba_array[0]) > 1:
                phishing_proba = proba_array[0][1]
                if np.isnan(phishing_proba):
                    phishing_proba = 0.0
                    reasons.append("Model returned an invalid risk score (NaN).")
            else:
                raise ValueError("Invalid output from model.predict_proba")
        except Exception as model_exc:
            reasons.append(f"Model prediction failed: {type(model_exc).__name__}. Defaulting to SAFE.")
            phishing_proba = 0.0

        # Step 3: Initial classification based on thresholds
        initial_prediction = "SAFE"
        if phishing_proba > 0.92:
            initial_prediction = "PHISHING"
        elif phishing_proba > 0.70:
            initial_prediction = "SUSPICIOUS"

        # Step 4: Add URL-based heuristic reasons
        if feats['suspicious_word_count'] > 0:
            reasons.append(f"Contains suspicious keywords: {', '.join(found_words)}")
        if feats['has_ip']:
            reasons.append("URL uses an IP address instead of a domain name.")
        if feats['subdomain_count'] > 2:
            reasons.append(f"URL has a high number of subdomains ({feats['subdomain_count']}).")
        if feats['has_https'] == 0:
             reasons.append("Site is not using HTTPS (unencrypted).")

        # Step 5: Get domain age (with robust error handling)
        domain = urlparse(req.url).netloc
        age_days = None
        is_registered = False
        try:
            w = whois.whois(domain)
            creation_date = w.creation_date
            if isinstance(creation_date, list):
                creation_date = creation_date[0]
            if creation_date:
                if creation_date.tzinfo is not None:
                    creation_date = creation_date.replace(tzinfo=None)
                age_days = (datetime.now() - creation_date).days
                is_registered = True
        except Exception:
            pass # We will add the reason in the next step if age is None

        # Step 6: Add age-related reasons
        if age_days is not None:
            if age_days < 30:
                reasons.append(f"Domain is extremely new ({age_days} days old).")
            elif age_days < 180:
                reasons.append(f"Domain is relatively new ({age_days} days old).")
        else:
            reasons.append("Could not determine domain age (WHOIS lookup failed).")

        # Step 7: Apply final override rule
        final_prediction = initial_prediction
        if age_days and age_days > 365 and initial_prediction in ["PHISHING", "SUSPICIOUS"]:
            final_prediction = "SAFE"
            
        # Step 8: Adjust reasons for logical consistency
        if final_prediction == "SAFE":
            if age_days and age_days > 365 and initial_prediction in ["PHISHING", "SUSPICIOUS"]:
                reasons = [f"Override: Marked as SAFE due to very old domain age ({age_days} days)."]
            elif len(reasons) > 0:
                reasons.insert(0, "Overall analysis determined the site to be SAFE despite some minor risk factors.")
            else:
                if phishing_proba < 0.10:
                    reasons = ["No specific risk factors found."]
                else:
                    reasons = ["Model score is low, but not zero. Considered SAFE after overall analysis."]
        elif not reasons: 
             reasons.append("Model indicated high risk, but no clear heuristic factors were identified.")

        # Step 9: Formulate response
        result = {
            "url": req.url,
            "prediction": final_prediction,
            "risk_score": float(round(phishing_proba * 100, 2)),
            "domain_age_days": age_days,
            "registered": is_registered,
            "reasons": reasons
        }
        
        return result

    except Exception as e:
        return {"error": str(e)}

@app.post("/check_age")
def check_domain_age(req: DomainRequest):
    domain = req.domain.lower().strip()
    try:
        w = whois.whois(domain)
        creation_date = w.creation_date
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
        if not creation_date:
            return {"domain_age_days": None, "registered": False}
        if creation_date.tzinfo is not None:
            creation_date = creation_date.replace(tzinfo=None)
        age_days = (datetime.now() - creation_date).days
        return {"domain_age_days": age_days, "registered": True}
    except Exception as e:
        return {"domain_age_days": None, "registered": False, "error": str(e)}
