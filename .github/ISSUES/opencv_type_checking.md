# OpenCV Type Checking Issues in Image Processing Tools

## Issue Description
Type checking errors have been identified in our image processing tools, specifically in the handling of OpenCV types and their interaction with numpy arrays. These issues affect the type safety of our codebase while not impacting runtime functionality.

## Current Problems

### 1. `fillPoly` Type Mismatch
**Location**: `src/helixzone/core/tools.py:142`
```python
cv2.fillPoly(mat, [points_array], color, lineType=line_type)
```
**Error**: 
```
Argument of type "NDArray[uint8]" cannot be assigned to parameter "img" of type "Mat"
```

### 2. `GaussianBlur` Type Incompatibilities
**Locations**: 
- `src/helixzone/core/tools.py:970`
- `src/helixzone/core/tools.py:1551`
- `src/helixzone/core/tools.py:2148`

**Error Pattern**:
```
Argument of type "ndarray[dtype[floating[_32Bit]]]" cannot be assigned to parameter "src" 
of type "NDArray[_TImg@GaussianBlur]"
```

## Impact
- No runtime errors or functionality issues
- Type checker warnings affect code quality metrics
- Potential maintenance challenges
- IDE support may be limited

## Current Workarounds
1. Using `# type: ignore` comments
2. Custom type conversion functions:
   - `ensure_mat`
   - `ensure_float32`
   - `ensure_uint8`

## Root Causes
1. OpenCV's Python bindings lack proper type hints
2. Complex type relationships between OpenCV and numpy arrays
3. Generic type constraints in OpenCV functions

## Proposed Solutions

### Short-term
1. Add targeted type ignores with explanatory comments
2. Document type conversion patterns
3. Create helper functions for common conversions

### Long-term
1. Develop comprehensive type stubs for OpenCV
2. Contribute type definitions upstream
3. Implement runtime type validation

## Related Files
- `src/helixzone/core/tools.py`
- `src/helixzone/core/type_defs.py`

## Next Steps
1. [ ] Create type stub file for core OpenCV functions
2. [ ] Update type conversion utilities
3. [ ] Add documentation for type handling
4. [ ] Consider contributing to OpenCV's type system

## References
- [OpenCV-Python Types Issue](https://github.com/opencv/opencv/issues/20997)
- [Numpy Type Annotations](https://numpy.org/devdocs/reference/typing.html)
- [Python Type Hints PEP 484](https://www.python.org/dev/peps/pep-0484/) 