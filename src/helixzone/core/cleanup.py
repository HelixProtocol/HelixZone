"""Cleanup and performance reporting module for HelixZone application."""

from typing import Dict, Any, List
import json
import os
from datetime import datetime
from pathlib import Path
import psutil
from .debugger import ProjectDebugger

class CleanupManager:
    """Manages application cleanup and generates performance reports."""
    
    def __init__(self, debugger: ProjectDebugger):
        """Initialize cleanup manager.
        
        Args:
            debugger: ProjectDebugger instance
        """
        self.debugger = debugger
        self.report_dir = Path(debugger.log_dir) / "reports"
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report.
        
        Returns:
            Dictionary containing performance metrics
        """
        perf_report = self.debugger.get_performance_report()
        error_report = self.debugger.get_error_report()
        process = psutil.Process()
        
        # Format performance metrics
        formatted_report = {
            'system_info': {
                'session_duration': (datetime.now() - self.debugger.start_time).total_seconds(),
                'final_memory_mb': process.memory_info().rss / 1024 / 1024,
                'final_cpu_percent': process.cpu_percent(),
                'total_errors': error_report['total_errors']
            },
            'function_metrics': {},
            'error_analysis': {
                'error_counts': error_report['error_counts'],
                'error_locations': error_report['error_locations']
            },
            'checkpoints': self.debugger.checkpoints
        }
        
        # Format function-specific metrics
        for func_name, metrics in perf_report.items():
            formatted_report['function_metrics'][func_name] = {
                'avg_time': f"{metrics['avg_time']:.4f}s",
                'min_time': f"{metrics['min_time']:.4f}s",
                'max_time': f"{metrics['max_time']:.4f}s",
                'total_time': f"{metrics['total_time']:.4f}s",
                'calls': metrics['calls'],
                'success_rate': f"{((metrics['calls'] - error_report['error_counts'].get(func_name, 0)) / metrics['calls'] * 100):.1f}%" if metrics['calls'] > 0 else "N/A"
            }
            
        return formatted_report
    
    def save_performance_report(self, report: Dict[str, Any]) -> str:
        """Save performance report to file.
        
        Args:
            report: Performance report dictionary
            
        Returns:
            Path to saved report file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.report_dir / f"performance_report_{timestamp}.json"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
            
        return str(report_path)
    
    def print_performance_summary(self, report: Dict[str, Any]) -> None:
        """Print formatted performance summary to console.
        
        Args:
            report: Performance report dictionary
        """
        print("\nPerformance Summary:")
        print("=" * 80)
        
        # Print system info
        sys_info = report['system_info']
        print("\nSystem Information:")
        print(f"  Session Duration: {sys_info['session_duration']:.2f}s")
        print(f"  Final Memory Usage: {sys_info['final_memory_mb']:.2f}MB")
        print(f"  Final CPU Usage: {sys_info['final_cpu_percent']:.1f}%")
        print(f"  Total Errors: {sys_info['total_errors']}")
        
        # Print function metrics
        print("\nFunction Performance:")
        for func_name, metrics in report['function_metrics'].items():
            print(f"\n{func_name}:")
            print(f"  Average Time: {metrics['avg_time']}")
            print(f"  Min Time: {metrics['min_time']}")
            print(f"  Max Time: {metrics['max_time']}")
            print(f"  Total Time: {metrics['total_time']}")
            print(f"  Total Calls: {metrics['calls']}")
            print(f"  Success Rate: {metrics['success_rate']}")
        
        # Print error summary
        if report['error_analysis']['error_counts']:
            print("\nError Summary:")
            for section, count in report['error_analysis']['error_counts'].items():
                print(f"  {section}: {count} errors")
            
            print("\nTop Error Locations:")
            sorted_locations = sorted(
                report['error_analysis']['error_locations'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]  # Show top 5 error locations
            for location, count in sorted_locations:
                print(f"  {location}: {count} errors")
        
        # Print checkpoint summary
        if report['checkpoints']:
            print("\nKey Checkpoints:")
            for checkpoint in report['checkpoints'][-5:]:  # Show last 5 checkpoints
                elapsed = checkpoint['elapsed']
                memory = checkpoint.get('memory_mb', 'N/A')
                print(f"  {checkpoint['name']} - {elapsed:.2f}s - {memory}MB")
    
    def cleanup(self) -> None:
        """Perform cleanup and generate final reports."""
        try:
            # Generate performance report
            perf_report = self.generate_performance_report()
            
            # Save report
            report_path = self.save_performance_report(perf_report)
            
            # Print summary
            self.print_performance_summary(perf_report)
            print(f"\nDetailed performance report saved to: {report_path}")
            
        finally:
            # Always call debugger cleanup
            self.debugger.cleanup() 