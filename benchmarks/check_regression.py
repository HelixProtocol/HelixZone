"""
Check for performance regressions in benchmark results.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy import stats

class RegressionChecker:
    def __init__(self):
        self.thresholds = {
            "cpu_time": 1.15,      # 15% slower
            "gpu_time": 1.15,      # 15% slower
            "ram_usage": 1.20,     # 20% more memory
            "gpu_usage": 1.20,     # 20% more GPU memory
            "color_overhead": 1.25, # 25% more overhead
            "batch_speedup": 0.85   # 15% less speedup
        }
        self.min_samples = 3  # Minimum samples needed for trend analysis
        self.z_threshold = 2.0  # Z-score threshold for outlier detection
        
    def load_latest_results(self) -> Dict[str, float]:
        """Load the most recent benchmark results."""
        results_file = Path("benchmarks/results.md")
        if not results_file.exists():
            raise FileNotFoundError("No benchmark results found")
            
        content = results_file.read_text()
        results = {}
        
        # Parse 4K image results
        pattern = r"\|\s*4096x4096\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
        if match := re.search(pattern, content):
            results["cpu_time"] = float(match.group(1))
            results["gpu_time"] = float(match.group(2))
            results["ram_usage"] = float(match.group(3))
            results["gpu_usage"] = float(match.group(4))
            
        # Parse color overhead
        if match := re.search(r"Overhead:\s*([\d.]+)%", content):
            results["color_overhead"] = float(match.group(1))
            
        # Parse batch speedup
        if match := re.search(r"\|\s*8\s*\|[^|]+\|[^|]+\|\s*([\d.]+)x\s*\|", content):
            results["batch_speedup"] = float(match.group(1))
            
        return results
        
    def load_historical_data(self) -> Dict[str, List[float]]:
        """Load historical benchmark data."""
        history = {}
        history_dir = Path("benchmarks/history")
        
        if not history_dir.exists():
            return {}
            
        for file in sorted(history_dir.glob("results_*.md")):
            content = file.read_text()
            
            # Parse 4K image results
            pattern = r"\|\s*4096x4096\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
            if match := re.search(pattern, content):
                history.setdefault("cpu_time", []).append(float(match.group(1)))
                history.setdefault("gpu_time", []).append(float(match.group(2)))
                history.setdefault("ram_usage", []).append(float(match.group(3)))
                history.setdefault("gpu_usage", []).append(float(match.group(4)))
                
            # Parse color overhead
            if match := re.search(r"Overhead:\s*([\d.]+)%", content):
                history.setdefault("color_overhead", []).append(float(match.group(1)))
                
            # Parse batch speedup
            if match := re.search(r"\|\s*8\s*\|[^|]+\|[^|]+\|\s*([\d.]+)x\s*\|", content):
                history.setdefault("batch_speedup", []).append(float(match.group(1)))
                
        return history
        
    def check_regression(self, metric: str, current: float, history: List[float]) -> Tuple[bool, str]:
        """Check if current value indicates a regression compared to historical data."""
        if len(history) < self.min_samples:
            return False, f"Not enough historical data for {metric} (need {self.min_samples})"
            
        # Calculate baseline statistics
        baseline_mean = float(np.mean(history))
        baseline_std = float(np.std(history))
        
        # Calculate z-score
        z_score = float((current - baseline_mean) / baseline_std if baseline_std > 0 else 0)
        
        # Check against threshold
        threshold = self.thresholds[metric]
        if metric == "batch_speedup":
            # For speedup, lower is worse
            is_regression = bool(current < baseline_mean * threshold)
        else:
            # For other metrics, higher is worse
            is_regression = bool(current > baseline_mean * threshold)
            
        message = (
            f"{metric}: current={current:.3f}, "
            f"baseline={baseline_mean:.3f}±{baseline_std:.3f}, "
            f"z-score={z_score:.2f}"
        )
        
        return is_regression, message
        
    def check_all_regressions(self) -> Tuple[bool, List[str]]:
        """Check all metrics for performance regressions."""
        current_results = self.load_latest_results()
        historical_data = self.load_historical_data()
        
        has_regression = False
        messages = []
        
        for metric in self.thresholds:
            if metric not in current_results or metric not in historical_data:
                messages.append(f"Missing data for {metric}")
                continue
                
            is_regression, message = self.check_regression(
                metric,
                current_results[metric],
                historical_data[metric]
            )
            
            if is_regression:
                has_regression = True
                messages.append(f"REGRESSION: {message}")
            else:
                messages.append(f"OK: {message}")
                
        return has_regression, messages

def main():
    checker = RegressionChecker()
    has_regression, messages = checker.check_all_regressions()
    
    # Write results
    output = "# Performance Regression Check\n\n"
    for message in messages:
        output += f"- {message}\n"
        
    Path("benchmarks/regression_check.md").write_text(output)
    
    # Exit with error if regression detected
    if has_regression:
        print("Performance regression detected!")
        print("\n".join(messages))
        exit(1)
    else:
        print("No performance regressions detected.")
        exit(0)

if __name__ == "__main__":
    main() 