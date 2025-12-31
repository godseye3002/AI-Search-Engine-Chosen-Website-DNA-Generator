import requests
import time

response = requests.post(
    "http://localhost:8000/process",
    json={
        "data_source": "perplexity",
        "product_id": f"test-perplexity-{int(time.time())}"
    }
)

print(f"Status: {response.status_code}")
print(f"Response: {response.text}")
