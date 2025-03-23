"""Error analysis module with comprehensive reporting and visualization."""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import time
from collections import defaultdict
import json
from pathlib import Path
from datetime import datetime, timedelta
from ..core.debugger import ProjectDebugger

# Get the global debugger instance
debugger = ProjectDebugger(
    project_name="helixzone",
    log_dir="debug_logs"
)

@dataclass
class ErrorTrend:
    """Tracks error trends over time."""
    time_window: timedelta
    error_counts: Dict[str, List[int]] = field(default_factory=lambda: defaultdict(list))
    timestamps: List[float] = field(default_factory=list)

@dataclass
class ErrorPattern:
    """Identifies common error patterns."""
    error_type: str
    occurrences: int = 0
    locations: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    stack_traces: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    related_vars: Dict[str, List[Any]] = field(default_factory=lambda: defaultdict(list))

class ErrorAnalyzer:
    """Analyzes and reports error patterns with detailed insights."""
    
    def __init__(self, time_window: timedelta = timedelta(hours=1)):
        self.debugger = debugger
        self.time_window = time_window
        self.trends = ErrorTrend(time_window)
        self.patterns: Dict[str, ErrorPattern] = {}
        
    @debugger.trace_function
    def analyze_errors(self, detailed: bool = True) -> Dict[str, Any]:
        """Analyze error patterns and generate comprehensive report.
        
        Args:
            detailed: Whether to include detailed analysis
            
        Returns:
            Dictionary containing error analysis results
        """
        with self.debugger.error_boundary("error_analysis"):
            try:
                # Get error report
                error_report = self.debugger.get_error_report()
                
                # Log analysis start
                self.debugger.checkpoint("Starting error analysis")
                
                # Analyze patterns
                self._analyze_patterns(error_report)
                
                # Generate report
                report = self._generate_report(error_report, detailed)
                
                # Log report
                self.debugger.state_logger.info(
                    f"Error analysis completed: {json.dumps(report, indent=2)}"
                )
                
                return report
                
            except Exception as e:
                self.debugger.error_logger.error(
                    f"Error analysis failed: {str(e)}"
                )
                raise
            finally:
                self.debugger.checkpoint("Error analysis completed")
    
    def _analyze_patterns(self, error_report: Dict[str, Any]) -> None:
        """Analyze error patterns from the error report.
        
        Args:
            error_report: Error report from debugger
        """
        # Update trends
        current_time = time.time()
        self.trends.timestamps.append(current_time)
        
        for section, count in error_report['error_counts'].items():
            self.trends.error_counts[section].append(count)
        
        # Clean old trend data
        cutoff_time = current_time - self.time_window.total_seconds()
        while self.trends.timestamps and self.trends.timestamps[0] < cutoff_time:
            self.trends.timestamps.pop(0)
            for counts in self.trends.error_counts.values():
                if counts:
                    counts.pop(0)
        
        # Analyze patterns
        for section, count in error_report['error_counts'].items():
            if section not in self.patterns:
                self.patterns[section] = ErrorPattern(error_type=section)
            
            pattern = self.patterns[section]
            pattern.occurrences += count
            
            # Update location counts
            for location, loc_count in error_report['error_locations'].items():
                pattern.locations[location] += loc_count
    
    def _generate_report(
        self,
        error_report: Dict[str, Any],
        detailed: bool
    ) -> Dict[str, Any]:
        """Generate comprehensive error analysis report.
        
        Args:
            error_report: Error report from debugger
            detailed: Whether to include detailed analysis
            
        Returns:
            Dictionary containing analysis results
        """
        report = {
            'summary': {
                'total_errors': error_report['total_errors'],
                'unique_sections': len(error_report['error_counts']),
                'unique_locations': len(error_report['error_locations']),
                'timestamp': datetime.now().isoformat()
            },
            'error_counts': self._format_error_counts(error_report),
            'error_locations': self._format_error_locations(error_report),
            'trends': self._calculate_trends()
        }
        
        if detailed:
            report.update({
                'patterns': self._analyze_error_patterns(),
                'recommendations': self._generate_recommendations()
            })
        
        return report
    
    def _format_error_counts(
        self,
        error_report: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Format error counts for reporting.
        
        Args:
            error_report: Error report from debugger
            
        Returns:
            List of formatted error counts
        """
        counts = []
        for section, count in error_report['error_counts'].items():
            counts.append({
                'section': section,
                'count': count,
                'percentage': (count / error_report['total_errors'] * 100)
                if error_report['total_errors'] > 0 else 0
            })
        return sorted(counts, key=lambda x: x['count'], reverse=True)
    
    def _format_error_locations(
        self,
        error_report: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Format error locations for reporting.
        
        Args:
            error_report: Error report from debugger
            
        Returns:
            List of formatted error locations
        """
        locations = []
        total_errors = sum(error_report['error_locations'].values())
        
        for location, count in error_report['error_locations'].items():
            file_path = Path(location.split(':')[0])
            line_number = int(location.split(':')[1])
            
            locations.append({
                'file': str(file_path),
                'line': line_number,
                'count': count,
                'percentage': (count / total_errors * 100)
                if total_errors > 0 else 0
            })
        return sorted(locations, key=lambda x: x['count'], reverse=True)
    
    def _calculate_trends(self) -> Dict[str, Any]:
        """Calculate error trends over time.
        
        Returns:
            Dictionary containing trend analysis
        """
        trends = {}
        
        for section, counts in self.trends.error_counts.items():
            if counts:
                trends[section] = {
                    'current': counts[-1],
                    'previous': counts[0] if len(counts) > 1 else 0,
                    'change_percentage': (
                        ((counts[-1] - counts[0]) / counts[0] * 100)
                        if len(counts) > 1 and counts[0] > 0
                        else 0
                    )
                }
        
        return trends
    
    def _analyze_error_patterns(self) -> Dict[str, Any]:
        """Analyze error patterns for insights.
        
        Returns:
            Dictionary containing pattern analysis
        """
        patterns = {}
        
        for section, pattern in self.patterns.items():
            patterns[section] = {
                'occurrences': pattern.occurrences,
                'top_locations': sorted(
                    pattern.locations.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5],
                'frequency': pattern.occurrences / len(self.trends.timestamps)
                if self.trends.timestamps else 0
            }
        
        return patterns
    
    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate error handling recommendations.
        
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Analyze high-frequency errors
        for section, pattern in self.patterns.items():
            if pattern.occurrences > 10:  # Threshold for high-frequency
                recommendations.append({
                    'type': 'high_frequency',
                    'section': section,
                    'occurrences': pattern.occurrences,
                    'suggestion': f"Consider adding specific error handling for {section}"
                })
        
        # Analyze error locations
        for section, pattern in self.patterns.items():
            top_locations = sorted(
                pattern.locations.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            if top_locations:
                recommendations.append({
                    'type': 'error_hotspot',
                    'section': section,
                    'locations': top_locations,
                    'suggestion': "Review error handling in these locations"
                })
        
        return recommendations
    
    def print_analysis(self, detailed: bool = True) -> None:
        """Print formatted error analysis to console.
        
        Args:
            detailed: Whether to include detailed analysis
        """
        report = self.analyze_errors(detailed)
        
        print("\nError Analysis Report")
        print("=" * 50)
        
        # Print summary
        print("\nSummary:")
        print(f"Total Errors: {report['summary']['total_errors']}")
        print(f"Unique Error Sections: {report['summary']['unique_sections']}")
        print(f"Unique Error Locations: {report['summary']['unique_locations']}")
        
        # Print error counts
        print("\nError Counts by Section:")
        for error in report['error_counts']:
            print(f"{error['section']}: {error['count']} "
                  f"({error['percentage']:.1f}%)")
        
        # Print error locations
        print("\nTop Error Locations:")
        for location in report['error_locations'][:5]:  # Show top 5
            print(f"{location['file']}:{location['line']} - "
                  f"{location['count']} errors ({location['percentage']:.1f}%)")
        
        if detailed:
            # Print trends
            print("\nError Trends:")
            for section, trend in report['trends'].items():
                print(f"{section}:")
                print(f"  Current: {trend['current']}")
                print(f"  Change: {trend['change_percentage']:.1f}%")
            
            # Print recommendations
            print("\nRecommendations:")
            for rec in report['recommendations']:
                print(f"- {rec['suggestion']}")
                if rec['type'] == 'error_hotspot':
                    for loc, count in rec['locations']:
                        print(f"  * {loc}: {count} errors")

# Create an analyzer instance
analyzer = ErrorAnalyzer()

# Get and print error analysis
analyzer.print_analysis() 