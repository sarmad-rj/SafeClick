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
        
        # Step 1: Extract features and generate initial heuristic findings
        feats, found_words = extract_features(req.url)
        if not feats:
            return {"error": "Invalid URL for feature extraction"}
        
        # Create a DataFrame with feature names to avoid UserWarning
        X = pd.DataFrame([feats], columns=feature_list)
        
        # Step 2: Get phishing probability from model
        phishing_proba = model.predict_proba(X)[0][1]

        # Step 3: Initial classification based on notebook logic thresholds
        initial_prediction = "SAFE"
        if phishing_proba > 0.92:
            initial_prediction = "PHISHING"
        elif phishing_proba > 0.70:
            initial_prediction = "SUSPICIOUS"

        # Step 4: Get domain age
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
        except:
            pass 

        # Step 5: Generate heuristic reasons (unconditionally based on features, but added to list based on 'initial_prediction' or for clarity)
        # Add URL-based heuristic reasons
        if feats['suspicious_word_count'] > 0:
            reasons.append(f"Contains suspicious keywords: {', '.join(found_words)}")
        if feats['has_ip']:
            reasons.append("URL uses an IP address instead of a domain name.")
        if feats['subdomain_count'] > 2: # Consider a threshold for subdomain count
            reasons.append(f"URL has a high number of subdomains ({feats['subdomain_count']}).")
        
        # Add domain age related heuristic reasons if applicable
        if age_days is not None:
            if age_days < 30:
                reasons.append(f"Domain is extremely new ({age_days} days old).")
            elif age_days < 180:
                reasons.append(f"Domain is relatively new ({age_days} days old).")
        else: # If age could not be determined, and it's not a safe prediction
            if initial_prediction in ["PHISHING", "SUSPICIOUS"]:
                reasons.append("Could not determine domain age (potentially unregistered or new TLD).")


        # Step 6: Apply final override rule
        final_prediction = initial_prediction
        if age_days and age_days > 365 and initial_prediction in ["PHISHING", "SUSPICIOUS"]:
            final_prediction = "SAFE"
            
        # Step 7: Adjust reasons based on final_prediction for logical consistency
        if final_prediction == "SAFE":
            # Case 1: Overridden to SAFE because of old age
            if age_days and age_days > 365 and initial_prediction in ["PHISHING", "SUSPICIOUS"]:
                reasons = [f"Override: Marked as SAFE due to very old domain age ({age_days} days)."]
            # Case 2: Some minor risks found, but model/age deemed it SAFE overall
            elif len(reasons) > 0:
                reasons.insert(0, "Overall analysis determined the site to be SAFE despite some minor risk factors.")
            # Case 3: No heuristic risks found
            else:
                # If score is negligible, it's truly clean
                if phishing_proba < 0.10:
                    reasons = ["No specific risk factors found."]
                # If score is not negligible, explain that the model had some concern
                else:
                    reasons = ["Model score is low, but not zero. Considered SAFE after overall analysis."]
        elif not reasons: 
             # Case 4: It's risky, but we couldn't find a clear heuristic reason why
             reasons.append("Model indicated high risk, but no clear heuristic factors were identified.")

        # Step 8: Formulate response
        result = {
            "url": req.url,
            "prediction": final_prediction,
            "risk_score": round(phishing_proba * 100, 2),
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
