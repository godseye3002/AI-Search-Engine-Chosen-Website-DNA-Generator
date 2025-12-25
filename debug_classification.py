#!/usr/bin/env python3
"""
Debug classification to see what's happening
"""

import sys
sys.path.append('.')

from classification_core import classify_website, is_obvious_special_url

def main():
    test_urls = [
        "https://mixpanel.com",
        "https://googleanalytics.com", 
        "https://segment.com"
    ]
    
    for url in test_urls:
        print(f"\n=== Testing URL: {url} ===")
        
        # Test obvious special URL check
        is_obvious = is_obvious_special_url(url)
        print(f"Is obvious special URL: {is_obvious}")
        
        # Test full classification
        input_data = {
            'url': url,
            'text': f'Test website {url}',
            'related_to': 'marketing analytics tools'
        }
        
        result = classify_website(input_data)
        print(f"Classification: {result.classification}")
        print(f"Error: {result.error}")
        if result.special_url_info:
            print(f"Special URL info: {result.special_url_info}")

if __name__ == "__main__":
    main()
