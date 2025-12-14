import pandas as pd
import joblib
import json
import whois
from datetime import datetime
from urllib.parse import urlparse
import re
import numpy as np

# This script is a standalone tester for the backend prediction logic.
# It uses the same model, features, and functions to mirror the backend's output.

# ---------------- Load model & features ----------------
try:
    model = joblib.load("phish_model_deploy.pkl")
    with open("feature_list.json", "r") as f:
        feature_list = json.load(f)
except FileNotFoundError as e:
    print(f"Error: Could not load required file '{e.filename}'.")
    print("Please make sure 'phish_model_deploy.pkl' and 'feature_list.json' are in the same directory.")
    exit()

# ---------------- Helper functions (copied from backend.py) ----------------
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

# Pydantic model mock for testing without FastAPI
class URLRequest:
    def __init__(self, url):
        self.url = url

# ---------------- Prediction Logic (copied from backend.py) ----------------
def predict(req: URLRequest):
    try:
        reasons = []
        
        # Step 1: Extract features and generate initial heuristic findings
        feats, found_words = extract_features(req.url)
        if not feats:
            return {"error": "Invalid URL for feature extraction"}
        
        # Add reasons for URL-based heuristics if found
        if feats['suspicious_word_count'] > 0:
            reasons.append(f"Contains suspicious keywords: {', '.join(found_words)}")
        if feats['has_ip']:
            reasons.append("URL uses an IP address instead of a domain name.")
        if feats['subdomain_count'] > 2:
            reasons.append(f"URL has a high number of subdomains ({feats['subdomain_count']}).")

        # Create a DataFrame with feature names to avoid UserWarning
        X = pd.DataFrame([feats], columns=feature_list)
        
        # Step 2: Get phishing probability from model
        phishing_proba = model.predict_proba(X)[0][1]

        # Step 3: Initial classification based on thresholds
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

        # Step 5: Add domain age-related reasons
        if age_days is not None:
            if age_days < 30:
                reasons.append(f"Domain is extremely new ({age_days} days old).")
            elif age_days < 180:
                reasons.append(f"Domain is relatively new ({age_days} days old).")
        else:
            if initial_prediction in ["PHISHING", "SUSPICIOUS"]:
                reasons.append("Could not determine domain age (potentially unregistered or new TLD).")

        # Step 6: Apply final override rule
        final_prediction = initial_prediction
        if age_days and age_days > 365 and initial_prediction in ["PHISHING", "SUSPICIOUS"]:
            final_prediction = "SAFE"
            
        # Step 7: Adjust reasons for logical consistency
        if final_prediction == "SAFE":
            if age_days and age_days > 365 and initial_prediction in ["PHISHING", "SUSPICIOUS"]:
                reasons = [f"Override: Marked as SAFE due to very old domain age ({age_days} days)."]
            elif len(reasons) > 0:
                reasons.insert(0, "Overall analysis determined the site to be SAFE despite some minor factors.")
            else:
                if phishing_proba < 0.10:
                    reasons = ["No specific risk factors found."]
                else:
                    reasons = ["Model score is low, but not zero. Considered SAFE after overall analysis."]
        elif not reasons: 
             reasons.append("Model indicated concern, but no clear heuristic risk factors were identified.")

        # Step 8: Formulate response
        result = {
            "url": req.url,
            "prediction": final_prediction,
            "confidence": round(phishing_proba * 100, 2),
            "domain_age_days": age_days,
            "registered": is_registered,
            "reasons": reasons
        }
        
        return result
    except Exception as e:
        return {"error": str(e)}

# ---------------- Main execution block ----------------
if __name__ == "__main__":
    # You can add or change URLs in this list to test them
    test_urls = [
        "https://github.com/login",
        "https://mail.google.com/mail/u/0/#inbox",
        "http://update-your-bank-details.xyz", # Hypothetical new phishing site
        "https://one.google.com/ai",
        "https://www.wikipedia.org/",
        "http://127.0.0.1/dashboard/",
        "http://example.com/login"
    ]

    print("--- Starting Model Test ---")
    for url in test_urls:
        print(f"\n--- Testing URL: {url} ---")
        
        mock_req = URLRequest(url=url)
        result = predict(mock_req)
        
        if "error" in result:
            print(f"  Error: {result['error']}")
        else:
            print(f"  Prediction: {result['prediction']}")
            print(f"  Phishing Risk Score: {result['risk_score']}%")
            print(f"  Domain Age: {result['domain_age_days']} days")
            print(f"  Reasons:")
            for reason in result['reasons']:
                print(f"    - {reason}")
    print("\n--- Test Complete ---")
