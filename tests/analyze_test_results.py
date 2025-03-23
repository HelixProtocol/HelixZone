#!/usr/bin/env python
"""
Test Results Analyzer for HelixZone

This script analyzes test results from previous runs to identify patterns
in test failures, track test stability over time, and provide insights on 
testing improvements.

Usage:
    python tests/analyze_test_results.py [--results_dir DIRECTORY] [--output_file FILENAME]
"""

import os
import sys
import json
import glob
import argparse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from collections import defaultdict, Counter

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze test results for HelixZone")
    parser.add_argument("--results_dir", type=str, default=".pytest_results",
                      help="Directory containing pytest result XML files")
    parser.add_argument("--output_file", type=str, default="test_analysis.md",
                      help="Output file to write the analysis to")
    parser.add_argument("--days", type=int, default=30,
                      help="Number of days of history to analyze")
    return parser.parse_args()

def find_result_files(results_dir, days):
    """Find all pytest result files in the specified directory from the last N days."""
    result_files = []
    cutoff_date = datetime.now() - timedelta(days=days)
    
    # Ensure directory exists
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        return []
    
    # Find all XML result files
    for filepath in glob.glob(os.path.join(results_dir, "*.xml")):
        file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
        if file_time >= cutoff_date:
            result_files.append(filepath)
    
    return sorted(result_files)

def parse_junit_xml(xml_file):
    """Parse a JUnit XML file and extract test results."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    results = {
        'total': 0,
        'failures': 0,
        'errors': 0,
        'skipped': 0,
        'time': 0,
        'date': datetime.fromtimestamp(os.path.getmtime(xml_file)),
        'test_cases': []
    }
    
    # Get suite details
    for testsuite in root.findall('.//testsuite'):
        results['total'] += int(testsuite.get('tests', 0))
        results['failures'] += int(testsuite.get('failures', 0))
        results['errors'] += int(testsuite.get('errors', 0))
        results['skipped'] += int(testsuite.get('skipped', 0))
        results['time'] += float(testsuite.get('time', 0))
        
        # Get individual test case results
        for testcase in testsuite.findall('.//testcase'):
            case = {
                'name': testcase.get('name'),
                'classname': testcase.get('classname'),
                'time': float(testcase.get('time', 0)),
                'status': 'passed'
            }
            
            # Check for failures, errors, or skips
            failure = testcase.find('failure')
            error = testcase.find('error')
            skipped = testcase.find('skipped')
            
            if failure is not None:
                case['status'] = 'failed'
                case['message'] = failure.get('message', '')
            elif error is not None:
                case['status'] = 'error'
                case['message'] = error.get('message', '')
            elif skipped is not None:
                case['status'] = 'skipped'
                case['message'] = skipped.get('message', '')
            
            results['test_cases'].append(case)
    
    return results

def analyze_results(result_files):
    """Analyze the results from multiple test runs."""
    all_results = []
    
    for file in result_files:
        try:
            result = parse_junit_xml(file)
            all_results.append(result)
        except Exception as e:
            print(f"Error parsing {file}: {e}")
    
    # Sort by date
    all_results.sort(key=lambda x: x['date'])
    
    return all_results

def find_flaky_tests(all_results):
    """Identify tests that sometimes pass and sometimes fail (flaky tests)."""
    test_status = defaultdict(list)
    
    # Collect status for each test across runs
    for result in all_results:
        for test_case in result['test_cases']:
            test_id = f"{test_case['classname']}::{test_case['name']}"
            test_status[test_id].append(test_case['status'])
    
    # Find tests with mixed results
    flaky_tests = {}
    for test_id, statuses in test_status.items():
        status_counts = Counter(statuses)
        if len(status_counts) > 1 and 'passed' in status_counts:
            flaky_tests[test_id] = {
                'pass_rate': status_counts['passed'] / len(statuses),
                'runs': len(statuses),
                'statuses': dict(status_counts)
            }
    
    return flaky_tests

def find_slow_tests(all_results, threshold=1.0):
    """Identify tests that are consistently slow."""
    test_times = defaultdict(list)
    
    # Collect execution times for each test
    for result in all_results:
        for test_case in result['test_cases']:
            test_id = f"{test_case['classname']}::{test_case['name']}"
            test_times[test_id].append(test_case['time'])
    
    # Calculate average time for each test
    slow_tests = {}
    for test_id, times in test_times.items():
        avg_time = sum(times) / len(times)
        if avg_time > threshold:
            slow_tests[test_id] = {
                'avg_time': avg_time,
                'max_time': max(times),
                'min_time': min(times),
                'runs': len(times)
            }
    
    # Sort by average time (slowest first)
    return {k: v for k, v in sorted(slow_tests.items(), key=lambda x: x[1]['avg_time'], reverse=True)}

def calculate_test_trends(all_results):
    """Calculate trends in test metrics over time."""
    dates = []
    pass_rates = []
    total_counts = []
    execution_times = []
    
    for result in all_results:
        dates.append(result['date'])
        total = result['total']
        total_counts.append(total)
        passed = total - result['failures'] - result['errors']
        pass_rates.append(passed / total if total > 0 else 0)
        execution_times.append(result['time'])
    
    return {
        'dates': dates,
        'pass_rates': pass_rates,
        'total_counts': total_counts,
        'execution_times': execution_times
    }

def generate_report(all_results, flaky_tests, slow_tests, trends, output_file):
    """Generate a Markdown report with the analysis results."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# HelixZone Test Analysis Report\n\n")
        f.write(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        # Summary
        total_runs = len(all_results)
        if total_runs > 0:
            latest = all_results[-1]
            f.write("## Latest Test Summary\n\n")
            f.write(f"- **Date**: {latest['date'].strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- **Total Tests**: {latest['total']}\n")
            f.write(f"- **Passed**: {latest['total'] - latest['failures'] - latest['errors']}\n")
            f.write(f"- **Failed**: {latest['failures']}\n")
            f.write(f"- **Errors**: {latest['errors']}\n")
            f.write(f"- **Skipped**: {latest['skipped']}\n")
            f.write(f"- **Execution Time**: {latest['time']:.2f} seconds\n\n")
        
        # Flaky Tests
        f.write("## Flaky Tests\n\n")
        if flaky_tests:
            f.write("These tests show inconsistent results and should be investigated:\n\n")
            f.write("| Test | Pass Rate | Runs | Status Breakdown |\n")
            f.write("|------|-----------|------|------------------|\n")
            for test_id, info in flaky_tests.items():
                status_str = ", ".join([f"{status}: {count}" for status, count in info['statuses'].items()])
                f.write(f"| {test_id} | {info['pass_rate']:.1%} | {info['runs']} | {status_str} |\n")
        else:
            f.write("No flaky tests detected in the analyzed period. 👍\n")
        
        # Slow Tests
        f.write("\n## Slow Tests\n\n")
        if slow_tests:
            f.write("These tests consistently take a long time to execute and might benefit from optimization:\n\n")
            f.write("| Test | Avg Time (s) | Max Time (s) | Min Time (s) | Runs |\n")
            f.write("|------|--------------|--------------|--------------|------|\n")
            for test_id, info in list(slow_tests.items())[:10]:  # Show top 10
                f.write(f"| {test_id} | {info['avg_time']:.2f} | {info['max_time']:.2f} | {info['min_time']:.2f} | {info['runs']} |\n")
            
            if len(slow_tests) > 10:
                f.write(f"\n*...and {len(slow_tests) - 10} more slow tests.*\n")
        else:
            f.write("No significantly slow tests detected in the analyzed period.\n")
        
        # Trends
        f.write("\n## Test Trends\n\n")
        if trends['dates']:
            start_date = trends['dates'][0].strftime('%Y-%m-%d')
            end_date = trends['dates'][-1].strftime('%Y-%m-%d')
            avg_pass_rate = sum(trends['pass_rates']) / len(trends['pass_rates'])
            avg_time = sum(trends['execution_times']) / len(trends['execution_times'])
            
            f.write(f"Analysis period: {start_date} to {end_date}\n\n")
            f.write(f"- **Average Pass Rate**: {avg_pass_rate:.1%}\n")
            f.write(f"- **Average Execution Time**: {avg_time:.2f} seconds\n")
            
            # Trend direction
            if len(trends['pass_rates']) >= 2:
                pass_rate_change = trends['pass_rates'][-1] - trends['pass_rates'][0]
                time_change = trends['execution_times'][-1] - trends['execution_times'][0]
                
                if pass_rate_change > 0.05:
                    f.write("- **Pass Rate Trend**: 📈 Improving\n")
                elif pass_rate_change < -0.05:
                    f.write("- **Pass Rate Trend**: 📉 Declining\n")
                else:
                    f.write("- **Pass Rate Trend**: ➡️ Stable\n")
                
                if time_change < -avg_time * 0.1:
                    f.write("- **Execution Time Trend**: 🚀 Getting faster\n")
                elif time_change > avg_time * 0.1:
                    f.write("- **Execution Time Trend**: 🐢 Getting slower\n")
                else:
                    f.write("- **Execution Time Trend**: ➡️ Stable\n")
        else:
            f.write("Not enough data to analyze trends.\n")
        
        # Recommendations
        f.write("\n## Recommendations\n\n")
        recommendations = []
        
        if flaky_tests:
            recommendations.append("**Stabilize Flaky Tests**: Focus on fixing the tests with the lowest pass rates")
        
        if slow_tests:
            recommendations.append("**Optimize Slow Tests**: Look for performance improvements in the slowest tests")
        
        if trends['dates'] and len(trends['dates']) >= 2:
            if sum(trends['pass_rates']) / len(trends['pass_rates']) < 0.9:
                recommendations.append("**Improve Pass Rate**: Overall pass rate is below 90%, focus on increasing reliability")
            
            if trends['pass_rates'][-1] < trends['pass_rates'][0]:
                recommendations.append("**Investigate Declining Pass Rate**: Pass rate has decreased over time")
            
            if trends['execution_times'][-1] > trends['execution_times'][0] * 1.2:
                recommendations.append("**Address Slowing Tests**: Test execution is getting significantly slower")
        
        if not recommendations:
            recommendations.append("**Maintain Current Quality**: No major issues detected, continue current practices")
        
        for i, rec in enumerate(recommendations, 1):
            f.write(f"{i}. {rec}\n")
    
    print(f"Analysis report generated: {output_file}")

def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Find result files
    result_files = find_result_files(args.results_dir, args.days)
    
    if not result_files:
        print(f"No test result files found in {args.results_dir} from the past {args.days} days.")
        # Create empty directory structure if it doesn't exist
        os.makedirs(args.results_dir, exist_ok=True)
        # Generate empty report
        generate_report([], {}, {}, {'dates': [], 'pass_rates': [], 'total_counts': [], 'execution_times': []}, args.output_file)
        return
    
    # Parse and analyze results
    all_results = analyze_results(result_files)
    flaky_tests = find_flaky_tests(all_results)
    slow_tests = find_slow_tests(all_results)
    trends = calculate_test_trends(all_results)
    
    # Generate report
    generate_report(all_results, flaky_tests, slow_tests, trends, args.output_file)

if __name__ == "__main__":
    main() 