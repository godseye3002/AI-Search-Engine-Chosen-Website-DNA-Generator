#!/usr/bin/env python3
"""
Test script for the serverless API endpoints
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Health check: {response.status_code} - {response.json()}")
        return response.ok
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_process():
    """Test process endpoint"""
    try:
        payload = {
            "product_id": "12345678-1234-1234-1234-123456789abc",  # Valid UUID format
            "source": "google"
        }
        response = requests.post(
            f"{BASE_URL}/process",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        print(f"Process test: {response.status_code}")
        if response.ok:
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        else:
            print(f"Error: {response.text}")
        return response.ok
    except Exception as e:
        print(f"Process test failed: {e}")
        return False

def main():
    print("Testing GodsEye DNA Pipeline API...")
    print(f"Base URL: {BASE_URL}")
    print("-" * 50)
    
    # Test health
    if not test_health():
        print("❌ Health check failed - server may not be running")
        return
    
    # Test process
    print("\nTesting process endpoint...")
    test_process()

if __name__ == "__main__":
    main()
