import os
import sys
import traceback
from pathlib import Path

# Ensure repo root is importable when running from /testing
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from error_email_sender import _format_error_payload, generate_ai_error_message, send_ai_error_email


def main():
    print("=" * 70)
    print("Testing Gemini Error Email Agent")
    print("=" * 70)

    try:
        # Simulate a real pipeline-ish error
        def stage_2_call():
            raise RuntimeError("Gemini quota exceeded while generating DNA chunk analysis")

        stage_2_call()

    except Exception as e:
        stack = traceback.format_exc()

        metadata = {
            "product_id": "02f92e70-7b53-45b6-bdef-7ef36d8fc578",
            "data_source": "google",
            "run_id": "dna_google_02f92e70-7b53-45b6-bdef-7ef36d8fc578_1234567890",
            "job_id": "job_7",
            "stage": "stage_2_dna_analysis",
            "url": "https://example.com/best-tools",
            "extra": {
                "endpoint": "/process",
                "component": "DNAAnalysisCore._analyze_html_content",
            },
        }

        payload = _format_error_payload(
            error=e,
            error_context="While processing product DNA pipeline. Error occurred inside Stage-2 Gemini call.",
            metadata=metadata,
            stack_trace=stack,
        )

        print("\n--- Generated Payload ---")
        for k in [
            "server_name",
            "timestamp_utc",
            "timestamp_local",
            "product_id",
            "data_source",
            "run_id",
            "job_id",
            "stage",
            "error_type",
            "error_message",
        ]:
            print(f"{k}: {payload.get(k)}")

        print("\n--- AI Message (Plain Text) ---")
        ai_message = generate_ai_error_message(payload)
        print(ai_message)

        # Optional: actually send the email (requires SMTP_* vars)
        # Use: SEND_TEST_EMAIL=1 python testing\\test_error_email_agent.py
        if os.getenv("SEND_TEST_EMAIL") == "1":
            ok = send_ai_error_email(
                error=e,
                error_context=payload.get("context", ""),
                metadata=metadata,
                stack_trace=stack,
            )
            print(f"\nEmail sent: {ok}")
        else:
            print("\n(Skipping SMTP send; set SEND_TEST_EMAIL=1 to send a real email)")


if __name__ == "__main__":
    main()
