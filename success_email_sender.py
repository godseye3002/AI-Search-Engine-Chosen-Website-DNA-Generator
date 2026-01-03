import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime
from dotenv import load_dotenv
from typing import Any, Dict, Optional, Tuple

from database.supabase_manager import SupabaseDataManager

load_dotenv()

SERVER_NAME = "Deep Analysis Website DNA Extracter"


def _is_truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_smtp_config() -> Tuple[str, int, str, str, str]:
    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "465"))
    smtp_user = os.getenv("SMTP_USER") or ""
    smtp_pass = os.getenv("SMTP_PASS") or ""
    from_email = os.getenv("FROM_EMAIL", smtp_user) or smtp_user
    return smtp_host, smtp_port, smtp_user, smtp_pass, from_email


def _get_user_contact_by_product_id(product_id: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    db = SupabaseDataManager()

    product_resp = (
        db.client.table("products")
        .select("id, user_id, product_name, product_url")
        .eq("id", product_id)
        .limit(1)
        .execute()
    )

    if not product_resp.data:
        return None, None, None, None

    product = product_resp.data[0]
    user_id = product.get("user_id")
    product_name = product.get("product_name")
    product_url = product.get("product_url")

    if not user_id:
        return None, product_name, product_url, None

    user_resp = (
        db.client.table("user_profiles")
        .select("id, user_name, email")
        .eq("id", user_id)
        .limit(1)
        .execute()
    )

    if not user_resp.data:
        return None, product_name, product_url, None

    user = user_resp.data[0]
    user_email = user.get("email")
    user_name = user.get("user_name")

    return user_email, product_name, product_url, user_name


def send_success_email_to_user(
    *,
    product_id: str,
    data_source: str,
    analysis_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> bool:
    if not _is_truthy(os.getenv("ENABLE_SUCCESS_EMAILS")):
        print(f"🔕 Success emails disabled - skipping email for product {product_id}")
        return False

    smtp_host, smtp_port, smtp_user, smtp_pass, from_email = _get_smtp_config()
    if not smtp_user or not smtp_pass:
        print(f"❌ SMTP configuration missing - cannot send success email for product {product_id}")
        return False

    user_email, product_name, product_url, user_name = _get_user_contact_by_product_id(product_id)
    if not user_email:
        print(f"❌ User email not found for product {product_id} - cannot send success email")
        return False

    print(f"📧 Sending success email to {user_email} for product {product_name} ({product_id})")

    subject = "✅ Your Website DNA Analysis is Complete"

    safe_product_name = product_name or "your product"
    safe_product_url = product_url or ""
    safe_user_name = user_name or "there"

    now_local = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = user_email

    html_body = f"""
    <div style=\"background:#f4f6fb;padding:40px 0;min-height:100vh;font-family:Arial,sans-serif;\">
        <div style=\"max-width:740px;margin:0 auto;background:#fff;border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,0.07);padding:28px 28px 20px 28px;\">
            <div style=\"font-size:20px;font-weight:700;color:#198754;margin-bottom:10px;\">{SERVER_NAME} - Analysis Complete</div>
            <div style=\"font-size:13px;color:#555;margin-bottom:18px;\">Local: {now_local}</div>

            <div style=\"font-size:14px;color:#222;margin-bottom:14px;\">Hi {safe_user_name},</div>

            <div style=\"background:#e7f3ff;border-left:4px solid #0d6efd;padding:14px;margin-bottom:14px;border-radius:4px;\">
                <div style=\"font-size:13px;color:#004085;\"><strong>Product:</strong> {safe_product_name}</div>
                {f'<div style="font-size:13px;color:#004085;margin-top:6px;"><strong>URL:</strong> <a href="{safe_product_url}">{safe_product_url}</a></div>' if safe_product_url else ''}
                <div style=\"font-size:13px;color:#004085;margin-top:6px;\"><strong>Source:</strong> {data_source}</div>
            </div>

            <div style=\"font-size:13px;color:#333;margin-bottom:10px;\">Your Website DNA analysis has completed successfully.</div>

            <div style=\"font-size:12px;color:#666;\">
                {f'<div><strong>Analysis ID:</strong> {analysis_id}</div>' if analysis_id else ''}
                {f'<div><strong>Run ID:</strong> {run_id}</div>' if run_id else ''}
            </div>

            <div style=\"text-align:center;font-size:12px;color:#aaa;margin-top:18px;\">— Automated Notification</div>
        </div>
    </div>
    """

    msg.attach(MIMEText(html_body, "html"))

    try:
        # Use SMTP_SSL for port 465, SMTP for other ports (like 587)
        if smtp_port == 465:
            print(f"🔐 Connecting to SMTP_SSL on port {smtp_port}")
            with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
                server.login(smtp_user, smtp_pass)
                server.send_message(msg)
        else:
            print(f"🔐 Connecting to SMTP with STARTTLS on port {smtp_port}")
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls()
                server.login(smtp_user, smtp_pass)
                server.send_message(msg)
        print(f"✅ Success email sent successfully to {user_email} for product {product_id}")
        return True
    except Exception as e:
        print(f"❌ Failed to send success email for product {product_id}: {e}")
        return False
