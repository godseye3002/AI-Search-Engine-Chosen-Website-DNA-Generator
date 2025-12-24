import requests
import json

# Configuration
API_URL = "http://localhost:8000/detect/batch"

def test_api():
    print(f"Sending request to {API_URL}...")

    # We construct a payload combining the Google and Perplexity examples.
    # We map the raw input keys to the API's expected 'DetectionInputModel' keys.
    
    payload = {
        "inputs": [
            # --- GOOGLE EXAMPLES ---
            # 1. A Google Search Link (Should be detected as 'special_url')
            {
                "url": "https://adjoe.io/blog/mobile-measurement-partner-mmp/#:~:text=A%20specialist%20in%20web%2Dto,Active%20Users%2C%20all%20features%20included.",
                "text": "The MMP Guide: Which Attribution Platform to Choose? - adjoe",
                "snippet": "9 Apr 2025 — A specialist in web-to-app measurement, Airbridge's strength is in-depth cohort-level analysis across web and mobile, in...",
                "position": 10,
                "related_to": "General"
            },
            {
                "url": "https://e-cens.com/blog/what-is-a-mobile-measurement-partner/#:~:text=down%20the%20road.-,Best%20Mobile%20Measurement%20Partners:%20Comparing%20Top%20MMPs,web%2Dto%2Dapp%20journeys.",
                "text": "What’s A Mobile Measurement Partner (MMP) - e-CENS",
                "snippet": "Best Mobile Measurement Partners: Comparing Top MMPs While the market has several players, a few consistently stand out as leaders...",
                "position": 9,
                "related_to": "General"
            },
            # {
            #     "text": "AppsFlyer",
            #     "url": "https://www.google.com/search?q=AppsFlyer&sca_esv=35aab6487a6cfb8f&source=hp&ei=9O48aaLSDeuN4-EP8NmtkQE&iflsig=AOw8s4IAAAAAaTz9BH2LlDtEj5pWffzhyPR8lT7M4kma&ved=2ahUKEwjRtouu4LmRAxXJT0EAHVtUMr0QgK4QegQIAxAB&uact=5&oq=what+are+the+best+MMP+app+tracking+analytics+tools+for+growth+marketers+needing+granular+channel+insights&gs_lp=Egdnd3Mtd2l6Iml3aGF0IGFyZSB0aGUgYmVzdCBNTVAgYXBwIHRyYWNraW5nIGFuYWx5dGljcyB0b29scyBmb3IgZ3Jvd3RoIG1hcmtldGVycyBuZWVkaW5nIGdyYW51bGFyIGNoYW5uZWwgaW5zaWdodHNI8StQvAJY0yhwAHgAkAEAmAHOBKABlhKqAQM1LTS4AQPIAQD4AQGYAgCgAgCoAgCYAwCSBwCgB9IDsgcAuAcAwgcAyAcAgAgA&sclient=gws-wiz&mstk=AUtExfBoMvMq5uRqxh8mbO_NGJa4PUCggTPE4AVAjqRoPatmY-ryCXljxMAkynJR46nDc5YTZdqIPkPknFWGnE-1ESpKD2ZQjpyjsVwH5oQT-bRpo4zI0RbGq-QRyKuA4xZ9_dI&csui=3",
            #     "related_claim": "Source was included in the AI search results and is related to Top MMPs for Granular Insights. Position: 1.",
            #     "position": 1
            # },
            # # 2. A Content Link found via Google (Should be processed normally)
            # {
            #     "text": "Top Mobile Measurement Partners in 2025 * AppsFlyer. AppsFlyer is your Mobile Measurement Partner (MMP)...",
            #     "url": "https://segwise.ai/blog/mobile-measurement-partners",
            #     "related_claim": "Source was included in the AI search results and is related to General. Position: 7.",
            #     "position": 7
            # },

            # --- PERPLEXITY EXAMPLES ---
            # 3. An informational article (Likely 'third_party')
            # {
            #     "text": "impact (+1 sources)",
            #     "url": "https://help.impact.com/en/support/solutions/articles/155000001524-mobile-tracking-using-mmps-explained",
            #     "raw_url": "https://help.impact.com/en/support/solutions/articles/155000001524-mobile-tracking-using-mmps-explained",
            #     "related_claim": ", or Mobile Measurement Partners, tracks link clicks and conversions primarily for mobile app campaigns...",
            #     "extraction_order": 1
            # },
            # # 4. A Product Home Page (Likely 'competitor' - should trigger HTML cleaning)
            # {
            #     "text": "singular",
            #     "url": "https://www.singular.net/mmp/",
            #     "raw_url": "https://www.singular.net/mmp/",
            #     "related_claim": "Singular: Provides universal linking across channels, partner integrations, and deep links for",
            #     "extraction_order": 3
            # }
        ]
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=120) # Increased timeout for batch processing
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"\n--- SUCCESS: Processed {data.get('total_processed')} items ---")
            print(f"Successful: {data.get('successful')}")
            print(f"Failed: {data.get('failed')}")
            
            # Pretty print the results
            print("\n--- DETAILED RESULTS ---")
            print(json.dumps(data['results'], indent=2))
            
            if data['errors']:
                print("\n--- ERRORS ---")
                print(json.dumps(data['errors'], indent=2))
        else:
            print(f"Error: Status Code {response.status_code}")
            print(response.text)

    except Exception as e:
        print(f"Connection Failed: {str(e)}")

if __name__ == "__main__":
    test_api()