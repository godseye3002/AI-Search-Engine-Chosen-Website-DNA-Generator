import subprocess
import sys

try:
    result = subprocess.run(
        [sys.executable, "test_perplexity_workflow.py"],
        capture_output=True,
        text=True,
        timeout=300
    )
    print("STDOUT:")
    print(result.stdout)
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    print(f"Return code: {result.returncode}")
except subprocess.TimeoutExpired:
    print("Test timed out after 5 minutes")
except Exception as e:
    print(f"Failed to run test: {e}")
