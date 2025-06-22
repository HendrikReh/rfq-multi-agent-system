#!/usr/bin/env python3
"""
Simple test runner for the Enhanced Multi-Agent RFQ System.

This script provides a convenient way to run all tests from the project root.
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Run the comprehensive test suite."""
    # Change to the tests directory and run the main test runner
    tests_dir = Path(__file__).parent / "tests"
    
    if not tests_dir.exists():
        print("❌ Tests directory not found!")
        sys.exit(1)
    
    try:
        # Run the test runner from the tests directory
        result = subprocess.run(
            [sys.executable, "run_all_tests.py"],
            cwd=tests_dir,
            capture_output=False
        )
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n👋 Test runner stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"💥 Error running tests: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 