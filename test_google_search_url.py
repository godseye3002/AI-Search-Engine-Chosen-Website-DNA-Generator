#!/usr/bin/env python3
"""
Test the Google search URL classification
"""

import sys
sys.path.append('.')

from classification_core import classify_website, is_obvious_special_url

def main():
    test_url = "https://www.google.com/search?q=HubSpot&sca_esv=cc24bdeaa2bd123f&source=hp&ei=H_9HaZrzDOvEkPIP4dLLuQU&iflsig=AOw8s4IAAAAAaUgNL8xTBLaB1KD6vnrsvSgM_k77-mFb&ved=2ahUKEwjRl8qr7c6RAxXRLrkGHXm6JpkQgK4QegQIBBAB&uact=5&oq=best+marketing+software+for+tracking+link+performance+and+optimizing+channels&gs_lp=Egdnd3Mtd2l6Ik1iZXN0IG1hcmtldGluZyBzb2Z0d2FyZSBmb3IgdHJhY2tpbmcgbGluayBwZXJmb3JtYW5jZSBhbmQgb3B0aW1pemluZyBjaGFubmVsc0iWPlDkAli1OnAAeACQAQCYAZoGoAHEGKoBAzYtNLgBA8gBAPgBAZgCAKACAKgCAJgDAJIHAKAHygOyBwC4BwDCBwDIBwCACAA&sclient=gws-wiz&mstk=AUtExfBMaiH1GjdqTIMf6Uix6WRSzUN5upApatDwdyN11ZlwrY-I20Bb_HBF6D6SCSnGbKmjhnOb44dch7vkPtiZQMFy12nifpr_wkSjDxG1lB3rwymHqsG9jMUQssKKWyb9dLg&csui=3"
    
    print(f"Testing Google search URL: {test_url[:100]}...")
    
    # Test obvious special URL check
    is_obvious = is_obvious_special_url(test_url)
    print(f"Is obvious special URL: {is_obvious}")
    
    # Test full classification
    input_data = {
        'url': test_url,
        'text': 'HubSpot',
        'related_to': 'marketing software'
    }
    
    result = classify_website(input_data)
    print(f"Classification: {result.classification}")
    print(f"Error: {result.error}")
    if result.special_url_info:
        print(f"Special URL info: {result.special_url_info}")

if __name__ == "__main__":
    main()
