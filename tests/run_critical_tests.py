#!/usr/bin/env python
"""
Run critical tests for HelixZone.

This script runs the critical tests for the HelixZone project, with options to:
- Run specific test files
- Set a timeout for tests
- Generate JUnit XML reports
- Generate coverage reports

Usage:
    python tests/run_critical_tests.py [options]

Options:
    --file FILE         Run a specific test file
    --verbose           Display verbose output
    --junit-xml         Generate JUnit XML reports
    --timeout SECONDS   Set a timeout for each test file (default: 30)
    --all               Run all critical tests
    --chunk             Run tests in smaller chunks to avoid timeouts
    --list              List all critical test files without running them
    --summary           Generate a test summary after running tests
    --help              Show help message
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Critical test files - these are considered essential to validate the core functionality
CRITICAL_TEST_FILES = [
    "tests/test_batch.py",
    "tests/test_image_processing.py",
    "tests/test_type_checker.py",
    "tests/test_color_processing.py::TestColorProcessingEdgeCases::test_solid_colors"
]

# Slow test files to skip by default
SLOW_TEST_FILES = [
    "tests/test_ml_utils.py",
    "tests/test_gpu.py",
    "tests/test_opencl.py"
]

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run critical tests for HelixZone")
    parser.add_argument("--file", help="Run a specific test file")
    parser.add_argument("--verbose", action="store_true", help="Display verbose output")
    parser.add_argument("--junit-xml", action="store_true", help="Generate JUnit XML reports")
    parser.add_argument("--timeout", type=int, default=30, help="Set a timeout for each test file (default: 30)")
    parser.add_argument("--all", action="store_true", help="Run all critical tests")
    parser.add_argument("--chunk", action="store_true", help="Run tests in smaller chunks to avoid timeouts")
    parser.add_argument("--list", action="store_true", help="List all critical test files without running them")
    parser.add_argument("--summary", action="store_true", help="Generate a test summary after running tests")
    
    return parser.parse_args()

def find_python_executable():
    """Find the Python executable to use for running tests."""
    # Try to use the same Python that's running this script
    python_exe = sys.executable
    
    # If we're in a virtual environment, use that Python
    virtual_env = os.environ.get('VIRTUAL_ENV')
    if virtual_env:
        if sys.platform == 'win32':
            python_exe = os.path.join(virtual_env, 'Scripts', 'python.exe')
        else:
            python_exe = os.path.join(virtual_env, 'bin', 'python')
    
    return python_exe

def run_test(test_file, python_exe, verbose=False, junit_xml=False, timeout=30):
    """Run a single test file."""
    print(f"Running {test_file}...")
    
    cmd = [python_exe, "-m", "pytest"]
    
    # Add options
    if verbose:
        cmd.append("-v")
    
    # Add coverage
    cmd.extend(["--cov=src"])
    
    # Add junit xml if requested
    if junit_xml:
        # Create the results directory if it doesn't exist
        os.makedirs(".pytest_results", exist_ok=True)
        
        # Create a filename from the test file path (replacing / with _)
        xml_filename = test_file.replace("/", "_").replace("\\", "_")
        # Add timestamp to avoid overwriting
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        xml_path = f".pytest_results/{xml_filename}_{timestamp}.xml"
        
        cmd.extend(["--junit-xml", xml_path])
    
    # Add the test file
    cmd.append(test_file)
    
    try:
        # Run the test with a timeout
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True,
            timeout=timeout
        )
        
        # Print the output
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        
        # Check the return code
        if result.returncode == 0:
            print(f"✅ Test {test_file} passed!")
            return True
        else:
            print(f"❌ Test {test_file} failed with code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏱️ Test {test_file} timed out after {timeout} seconds!")
        return False
    
    except Exception as e:
        print(f"❌ Error running test {test_file}: {e}")
        return False
    
    finally:
        print("-" * 80)

def main():
    """Main function."""
    args = parse_args()
    
    # Find the Python executable
    python_exe = find_python_executable()
    print(f"Running critical tests using Python: {python_exe}")
    
    if args.timeout:
        print(f"Timeout set to {args.timeout} seconds per test")
    
    # If --list is specified, just list the critical test files
    if args.list:
        print("Critical test files:")
        for test_file in CRITICAL_TEST_FILES:
            print(f"- {test_file}")
        return
    
    # Start tracking results
    passed = 0
    failed = 0
    timed_out = 0
    
    # Create results directory for junit xml if needed
    if args.junit_xml:
        os.makedirs(".pytest_results", exist_ok=True)
    
    # If --file is specified, run only that file
    if args.file:
        test_files = [args.file]
    elif args.all:
        # Run all tests in the tests directory
        test_files = glob.glob("tests/test_*.py")
    else:
        # Run only the critical tests
        test_files = CRITICAL_TEST_FILES
    
    print(f"Running {len(test_files)} critical tests using Python: {python_exe}")
    
    # Run the tests
    for test_file in test_files:
        if os.path.exists(test_file) or "::" in test_file:  # Allow for specific test selection
            result = run_test(
                test_file, 
                python_exe, 
                verbose=args.verbose, 
                junit_xml=args.junit_xml, 
                timeout=args.timeout
            )
            
            if result:
                passed += 1
            else:
                if "timed out" in test_file:
                    timed_out += 1
                else:
                    failed += 1
        else:
            print(f"❌ Test file {test_file} not found")
            failed += 1
    
    # Print results
    print("Critical test run complete!")
    print(f"Results: {passed} passed, {failed} failed, {timed_out} timed out")
    
    # Generate test summary if requested
    if args.summary:
        try:
            print("\nGenerating test analysis report...")
            # Find the summary script
            summary_script = os.path.join(os.path.dirname(__file__), "test_summary.py")
            
            if os.path.exists(summary_script):
                summary_cmd = [python_exe, summary_script]
                subprocess.run(summary_cmd, check=True)
            else:
                print(f"❌ Test summary script not found at {summary_script}")
        except Exception as e:
            print(f"Error generating test analysis: {e}")
    
    # Return appropriate exit code
    return 0 if failed == 0 and timed_out == 0 else 1

if __name__ == "__main__":
    sys.exit(main()) 