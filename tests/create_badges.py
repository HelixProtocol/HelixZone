#!/usr/bin/env python
"""
Badge generator script for HelixZone project.

This script analyzes test results and coverage reports to generate badges
that can be embedded in the project README.md or other documentation.
"""

import argparse
import json
import os
import sys
import subprocess
import re
from pathlib import Path
from datetime import datetime

# Badge templates with placeholders
BADGE_TEMPLATES = {
    "tests": "https://img.shields.io/badge/tests-{status}-{color}",
    "coverage": "https://img.shields.io/badge/coverage-{coverage}%25-{color}",
    "build": "https://img.shields.io/badge/build-{status}-{color}",
    "passing": "https://img.shields.io/badge/passing-{passing}%2F{total}-{color}"
}

# Colors for different statuses
COLORS = {
    "success": "brightgreen",
    "partial": "yellow",
    "failure": "red",
    "unknown": "lightgrey",
    # Coverage colors
    "high": "brightgreen",    # >= 80%
    "good": "green",          # >= 70%
    "moderate": "yellowgreen", # >= 60%
    "low": "yellow",          # >= 40%
    "poor": "orange",         # >= 20%
    "critical": "red"         # < 20%
}

def get_coverage_data(coverage_file="coverage.xml"):
    """Extract coverage data from the coverage report."""
    if not os.path.exists(coverage_file):
        return 0, "unknown", []
    
    try:
        # Extract coverage percentage from XML file using regex
        with open(coverage_file, 'r') as f:
            content = f.read()
            
        match = re.search(r'line-rate="([\d\.]+)"', content)
        if match:
            coverage = float(match.group(1)) * 100
        else:
            coverage = 0
            
        # Determine color based on coverage percentage
        if coverage >= 80:
            color = COLORS["high"]
        elif coverage >= 70:
            color = COLORS["good"]
        elif coverage >= 60:
            color = COLORS["moderate"]
        elif coverage >= 40:
            color = COLORS["low"]
        elif coverage >= 20:
            color = COLORS["poor"]
        else:
            color = COLORS["critical"]
            
        # Extract module data if available
        modules = []
        module_matches = re.finditer(r'<package name="(.+?)".*?line-rate="([\d\.]+)"', content)
        for match in module_matches:
            module_name = match.group(1)
            module_coverage = float(match.group(2)) * 100
            modules.append({
                "name": module_name,
                "coverage": module_coverage
            })
            
        return coverage, color, modules
            
    except Exception as e:
        print(f"Error processing coverage file: {e}")
        return 0, COLORS["unknown"], []

def get_test_results(results_dir=".pytest_results"):
    """Extract test results from JUnit XML files."""
    if not os.path.exists(results_dir):
        return 0, 0, COLORS["unknown"]
    
    try:
        total_tests = 0
        passed_tests = 0
        
        for xml_file in Path(results_dir).glob("*.xml"):
            with open(xml_file, 'r') as f:
                content = f.read()
                
            # Extract test counts
            tests_match = re.search(r'tests="(\d+)"', content)
            failures_match = re.search(r'failures="(\d+)"', content)
            errors_match = re.search(r'errors="(\d+)"', content)
            
            if tests_match:
                file_total = int(tests_match.group(1))
                total_tests += file_total
                
                file_failures = int(failures_match.group(1)) if failures_match else 0
                file_errors = int(errors_match.group(1)) if errors_match else 0
                file_passed = file_total - file_failures - file_errors
                passed_tests += file_passed
        
        # Determine color based on pass percentage
        if total_tests == 0:
            return 0, 0, COLORS["unknown"]
            
        pass_percentage = (passed_tests / total_tests) * 100
        
        if pass_percentage >= 90:
            color = COLORS["success"]
        elif pass_percentage >= 70:
            color = COLORS["partial"]
        else:
            color = COLORS["failure"]
            
        return passed_tests, total_tests, color
            
    except Exception as e:
        print(f"Error processing test results: {e}")
        return 0, 0, COLORS["unknown"]

def generate_badges(output_dir="badges"):
    """Generate badge URLs and save them to files."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get coverage data
    coverage, coverage_color, modules = get_coverage_data()
    
    # Get test results
    passed, total, test_color = get_test_results()
    
    # Generate badges
    badges = {
        "coverage": BADGE_TEMPLATES["coverage"].format(
            coverage=round(coverage, 1), 
            color=coverage_color
        ),
        "tests": BADGE_TEMPLATES["passing"].format(
            passing=passed, 
            total=total, 
            color=test_color
        ),
        "build": BADGE_TEMPLATES["build"].format(
            status="passing" if passed == total and total > 0 else "failing",
            color=COLORS["success"] if passed == total and total > 0 else COLORS["failure"]
        ),
        "timestamp": datetime.now().isoformat()
    }
    
    # Save badges to JSON file
    with open(os.path.join(output_dir, "badges.json"), 'w') as f:
        json.dump(badges, f, indent=2)
    
    # Save individual badge URLs
    for name, url in badges.items():
        if name != "timestamp":
            with open(os.path.join(output_dir, f"{name}.txt"), 'w') as f:
                f.write(url)
    
    # Print summary
    print(f"Coverage: {round(coverage, 1)}%")
    print(f"Tests: {passed}/{total} passing")
    print(f"Badges generated in: {output_dir}")
    
    return badges

def update_readme(badges, readme_path="README.md"):
    """Update README with badge URLs."""
    if not os.path.exists(readme_path):
        print(f"README file not found: {readme_path}")
        return False
    
    try:
        with open(readme_path, 'r') as f:
            content = f.read()
        
        # Replace badge URLs in README
        updated_content = content
        
        # Define badge markers in README
        markers = {
            "coverage": ("<!-- COVERAGE_BADGE_START -->", "<!-- COVERAGE_BADGE_END -->"),
            "tests": ("<!-- TESTS_BADGE_START -->", "<!-- TESTS_BADGE_END -->"),
            "build": ("<!-- BUILD_BADGE_START -->", "<!-- BUILD_BADGE_END -->"),
        }
        
        # Update each badge section
        for badge_name, (start_marker, end_marker) in markers.items():
            if start_marker in content and end_marker in content and badge_name in badges:
                badge_url = badges[badge_name]
                badge_md = f"![{badge_name.title()}]({badge_url})"
                pattern = f"{re.escape(start_marker)}.*?{re.escape(end_marker)}"
                replacement = f"{start_marker}{badge_md}{end_marker}"
                updated_content = re.sub(pattern, replacement, updated_content, flags=re.DOTALL)
        
        # Write updated content back to README
        with open(readme_path, 'w') as f:
            f.write(updated_content)
            
        print(f"README updated with badges: {readme_path}")
        return True
        
    except Exception as e:
        print(f"Error updating README: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Generate badges for test status and code coverage")
    parser.add_argument("--output-dir", default="badges", help="Directory to save badge files")
    parser.add_argument("--coverage-file", default="coverage.xml", help="Path to coverage XML file")
    parser.add_argument("--results-dir", default=".pytest_results", help="Directory containing test result XML files")
    parser.add_argument("--readme", default="README.md", help="Path to README.md file to update with badges")
    parser.add_argument("--update-readme", action="store_true", help="Update README with badges")
    
    args = parser.parse_args()
    
    badges = generate_badges(args.output_dir)
    
    if args.update_readme:
        update_readme(badges, args.readme)

if __name__ == "__main__":
    main() 