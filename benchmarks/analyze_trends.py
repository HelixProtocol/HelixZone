"""
Analyze benchmark trends over time and generate reports.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

class BenchmarkAnalyzer:
    def __init__(self):
        self.history_dir = Path("benchmarks/history")
        self.results: Dict[str, List[Tuple[datetime, float]]] = {
            "cpu_time": [],
            "gpu_time": [],
            "ram_usage": [],
            "gpu_usage": [],
            "color_overhead": [],
            "batch_speedup": []
        }
        
    def parse_date_from_filename(self, filename: str) -> datetime:
        """Extract date from benchmark result filename."""
        match = re.search(r"results_(\d{8})_(\d{6})", filename)
        if not match:
            raise ValueError(f"Invalid filename format: {filename}")
        date_str = f"{match.group(1)}_{match.group(2)}"
        return datetime.strptime(date_str, "%Y%m%d_%H%M%S")
        
    def parse_size_scaling(self, content: str) -> Optional[Dict[str, float]]:
        """Parse size scaling results from markdown content."""
        pattern = r"\|\s*4096x4096\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
        match = re.search(pattern, content)
        if not match:
            return None
        return {
            "cpu_time": float(match.group(1)),
            "gpu_time": float(match.group(2)),
            "ram_usage": float(match.group(3)),
            "gpu_usage": float(match.group(4))
        }
        
    def parse_color_overhead(self, content: str) -> Optional[float]:
        """Parse color processing overhead from markdown content."""
        pattern = r"Overhead:\s*([\d.]+)%"
        match = re.search(pattern, content)
        return float(match.group(1)) if match else None
        
    def parse_batch_speedup(self, content: str) -> Optional[float]:
        """Parse maximum batch processing speedup from markdown content."""
        pattern = r"\|\s*8\s*\|[^|]+\|[^|]+\|\s*([\d.]+)x\s*\|"
        match = re.search(pattern, content)
        return float(match.group(1)) if match else None
        
    def load_history(self) -> None:
        """Load and parse all historical benchmark results."""
        for file in sorted(self.history_dir.glob("results_*.md")):
            date = self.parse_date_from_filename(file.name)
            content = file.read_text()
            
            # Parse size scaling results (4K image performance)
            if scaling := self.parse_size_scaling(content):
                self.results["cpu_time"].append((date, scaling["cpu_time"]))
                self.results["gpu_time"].append((date, scaling["gpu_time"]))
                self.results["ram_usage"].append((date, scaling["ram_usage"]))
                self.results["gpu_usage"].append((date, scaling["gpu_usage"]))
            
            # Parse color processing overhead
            if overhead := self.parse_color_overhead(content):
                self.results["color_overhead"].append((date, overhead))
            
            # Parse batch processing speedup
            if speedup := self.parse_batch_speedup(content):
                self.results["batch_speedup"].append((date, speedup))
                
    def plot_trends(self) -> None:
        """Generate trend plots for each metric."""
        plt.style.use('seaborn')
        metrics = [
            ("Processing Time (4K Image)", ["cpu_time", "gpu_time"], "Time (s)"),
            ("Memory Usage (4K Image)", ["ram_usage", "gpu_usage"], "Memory (MB)"),
            ("Color Processing Overhead", ["color_overhead"], "Overhead (%)"),
            ("Batch Processing Speedup", ["batch_speedup"], "Speedup Factor")
        ]
        
        for title, keys, ylabel in metrics:
            plt.figure(figsize=(10, 6))
            for key in keys:
                dates, values = zip(*self.results[key])
                plt.plot(dates, values, marker='o', label=key.replace('_', ' ').title())
            
            plt.title(title)
            plt.xlabel("Date")
            plt.ylabel(ylabel)
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"benchmarks/trend_{keys[0]}.png")
            plt.close()
            
    def generate_report(self) -> None:
        """Generate a markdown report with trend analysis."""
        report = "# Benchmark Trend Analysis\n\n"
        
        # Latest vs. First measurements
        report += "## Performance Changes\n\n"
        report += "| Metric | First | Latest | Change |\n"
        report += "|--------|--------|---------|--------|\n"
        
        for key in self.results:
            if not self.results[key]:
                continue
            first = self.results[key][0][1]
            latest = self.results[key][-1][1]
            change = ((latest - first) / first) * 100
            
            report += f"| {key.replace('_', ' ').title()} "
            report += f"| {first:.3f} | {latest:.3f} | {change:+.1f}% |\n"
            
        # Add trend plots
        report += "\n## Trend Plots\n\n"
        for key in ["cpu_time", "ram_usage", "color_overhead", "batch_speedup"]:
            report += f"### {key.replace('_', ' ').title()}\n"
            report += f"![{key} trend](trend_{key}.png)\n\n"
            
        # Write report
        Path("benchmarks/trends.md").write_text(report)
        
    def analyze(self) -> None:
        """Run complete trend analysis."""
        self.load_history()
        self.plot_trends()
        self.generate_report()

if __name__ == "__main__":
    analyzer = BenchmarkAnalyzer()
    analyzer.analyze() 