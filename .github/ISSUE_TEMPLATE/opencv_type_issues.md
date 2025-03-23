---
name: OpenCV Type Checking Issues
about: Track and resolve OpenCV type compatibility problems in the codebase
title: 'Type System: OpenCV Type Checking Improvements Needed'
labels: 'type-system, opencv, enhancement'
assignees: ''
---

## Current Type Checking Issues

### 1. OpenCV Mat Type Compatibility
- `cv2.UMat` and `cv2.Mat` types lack proper type hints in Python
- Current workaround uses `# type: ignore` comments
- Affects functions like `fillPoly`, `GaussianBlur`, and other OpenCV operations

### 2. Specific Function Type Mismatches

#### `fillPoly` Issues:
```python
cv2.fillPoly(
    mat,  # Type error: NDArray[uint8] cannot be assigned to parameter "img" of type "Mat"
    [points_array],
    color,
    lineType=line_type
)
```

#### `GaussianBlur` Issues:
```python
# Multiple instances where ndarray[dtype[floating[_32Bit]]] is incompatible with NDArray[_TImg@GaussianBlur]
mask = cv2.GaussianBlur(mask, (0, 0), self.feather_radius)
```

### 3. Type Conversion Functions
Current type conversion functions need improvement:
```python
def ensure_mat(img: Union[NDArray[Any], cv2.UMat]) -> cv2.UMat
def ensure_float32(img: Union[NDArray[Any], cv2.UMat]) -> NDArray[np.float32]
def ensure_uint8(img: Union[NDArray[Any], cv2.UMat]) -> NDArray[np.uint8]
```

## Proposed Solutions

### Short-term:
1. Create type stubs for commonly used OpenCV functions
2. Implement proper type conversion utilities
3. Document type conversion patterns

### Long-term:
1. Create comprehensive type stubs for OpenCV-Python
2. Contribute type definitions upstream to OpenCV
3. Implement runtime type checking for critical conversions

## Implementation Plan

### Phase 1: Type Stub Creation
- [ ] Create basic type stubs for core OpenCV functions
- [ ] Define proper type hierarchies for Mat/UMat
- [ ] Add type variables for generic image types

### Phase 2: Type Conversion Utilities
- [ ] Implement robust type conversion functions
- [ ] Add runtime type checking where needed
- [ ] Create helper functions for common patterns

### Phase 3: Documentation
- [ ] Document type conversion patterns
- [ ] Add examples for common use cases
- [ ] Create troubleshooting guide

## Affected Files
- `src/helixzone/core/tools.py`
- `src/helixzone/core/type_defs.py`

## Additional Notes
- OpenCV's Python bindings use Numpy arrays internally
- Type checking needs to handle both OpenCV and Numpy types
- Consider performance implications of type conversions

## Resources
- [OpenCV Python Documentation](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)
- [Numpy Type Hints Documentation](https://numpy.org/devdocs/reference/typing.html)
- [PEP 484 – Type Hints](https://www.python.org/dev/peps/pep-0484/) 