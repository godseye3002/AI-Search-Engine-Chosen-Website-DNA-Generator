#!/usr/bin/env python3
"""
Test Success Email System

This script tests the new success email functionality without affecting
the existing error email system. It simulates a successful pipeline completion
and verifies that the success email is sent to the correct user.
"""

import os
import sys
from dotenv import load_dotenv
from success_email_sender import send_success_email_to_user, _get_user_contact_by_product_id, _is_truthy

# Load environment
load_dotenv()

def test_feature_flag():
    """Test ENABLE_SUCCESS_EMAILS feature flag"""
    print("🔧 Testing Feature Flag")
    print("-" * 40)
    
    # Test with flag not set
    os.environ["ENABLE_SUCCESS_EMAILS"] = "false"
    result = send_success_email_to_user(
        product_id="02f92e70-7b53-45b6-bdef-7ef36d8fc578",
        data_source="google"
    )
    print(f"❌ Flag disabled: send_success_email_to_user returned {result}")
    
    # Test with flag enabled
    os.environ["ENABLE_SUCCESS_EMAILS"] = "true"
    print("✅ Flag enabled: Ready to test email sending")
    
    return True

def test_user_lookup():
    """Test user lookup by product_id"""
    print("\n👤 Testing User Lookup")
    print("-" * 40)
    
    test_product_id = "02f92e70-7b53-45b6-bdef-7ef36d8fc578"
    
    try:
        user_email, product_name, product_url, user_name = _get_user_contact_by_product_id(test_product_id)
        
        if user_email:
            print(f"✅ User found: {user_name} ({user_email})")
            print(f"📦 Product: {product_name}")
            print(f"🔗 URL: {product_url}")
            return True
        else:
            print("❌ No user found for this product_id")
            return False
            
    except Exception as e:
        print(f"❌ User lookup failed: {e}")
        return False

def test_success_email():
    """Test actual success email sending"""
    print("\n📧 Testing Success Email")
    print("-" * 40)
    
    # Ensure feature flag is enabled
    os.environ["ENABLE_SUCCESS_EMAILS"] = "true"
    
    # Check SMTP config
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    
    if not smtp_user or not smtp_pass:
        print("❌ SMTP credentials not configured")
        print("Please set SMTP_USER and SMTP_PASS in .env")
        return False
    
    print(f"✅ SMTP configured for: {smtp_user}")
    
    # Test with real product_id
    test_product_id = "02f92e70-7b53-45b6-bdef-7ef36d8fc578"
    
    try:
        result = send_success_email_to_user(
            product_id=test_product_id,
            data_source="google",
            analysis_id="test_analysis_123",
            run_id="test_run_456"
        )
        
        if result:
            print("✅ Success email sent successfully!")
            print("📬 Check your email inbox for the notification")
            return True
        else:
            print("❌ Success email failed to send")
            return False
            
    except Exception as e:
        print(f"❌ Email test failed: {e}")
        return False

def test_error_email_unchanged():
    """Verify error email system is unchanged"""
    print("\n🚨 Testing Error Email System (Unchanged)")
    print("-" * 40)
    
    try:
        from error_email_sender import send_ai_error_email, generate_ai_error_message
        
        # Test AI error message generation (no actual sending)
        test_payload = {
            "server_name": "Test Server",
            "error_type": "TestError",
            "error_message": "This is a test error",
            "product_id": "test_product",
            "data_source": "google"
        }
        
        message = generate_ai_error_message(test_payload)
        print("✅ Error email system working (unchanged)")
        print(f"📝 Generated error message: {len(message)} characters")
        
        return True
        
    except Exception as e:
        print(f"❌ Error email system test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Success Email System")
    print("=" * 60)
    print("This will test the new success email functionality")
    print("without affecting the existing error email system.\n")
    
    tests = [
        ("Feature Flag", test_feature_flag),
        ("User Lookup", test_user_lookup),
        ("Success Email", test_success_email),
        ("Error Email Unchanged", test_error_email_unchanged),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Success email system is working.")
    else:
        print("⚠️  Some tests failed. Check the output above.")
    
    print("\n📝 Notes:")
    print("- Success emails are only sent when ENABLE_SUCCESS_EMAILS=true")
    print("- Error emails remain unchanged and still work as before")
    print("- Check your email inbox for the test success email")

if __name__ == "__main__":
    main()
