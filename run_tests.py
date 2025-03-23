import pytest
import sys

if __name__ == "__main__":
    # Add source directory to path
    sys.path.append("src")
    
    # Run tests with verbose output
    pytest.main(["-v", "tests/"]) 