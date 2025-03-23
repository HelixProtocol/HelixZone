# HelixZone UI Testing Plan

## Overview
This document outlines a comprehensive approach to automated UI testing for HelixZone, ensuring that the user interface components function correctly across platforms and configurations.

## Goals and Objectives

1. **Functionality Verification**: Verify that all UI components function as expected
2. **Usability Testing**: Ensure that the application is intuitive and user-friendly
3. **Cross-Platform Validation**: Confirm UI consistency across different operating systems
4. **Regression Prevention**: Detect UI regressions early in the development process
5. **Accessibility Testing**: Validate that the application is accessible to all users

## Testing Framework and Tools

### Core Testing Tools
- **pytest-qt**: Primary framework for testing PyQt applications
- **pytest-xvfb**: For headless UI testing in CI/CD environments
- **pytest-mock**: For mocking dependencies during testing
- **pytest-cov**: For measuring test coverage
- **screenshot-assertion**: For visual regression testing

### Additional Tools
- **Qt Test**: For low-level Qt event testing
- **QTest**: For simulating mouse and keyboard events
- **QSignalSpy**: For testing Qt signals
- **PyAutoGUI**: For cross-platform GUI automation (when needed)

## Test Types

### 1. Component Tests
Tests for individual UI components in isolation.

#### UI Components to Test
- Main window and its panels
- Toolbars and their actions
- Menu items and their actions
- Dialog boxes and forms
- Custom widgets (e.g., image view, layer panel)
- Status bar and its components

#### Example Component Test
```python
def test_toolbar_buttons(qtbot):
    """Test that toolbar buttons are created and respond to clicks."""
    # Setup
    toolbar = Toolbar()
    qtbot.addWidget(toolbar)
    
    # Verify initial state
    assert toolbar.isVisible()
    assert toolbar.actions()[0].text() == "Select"
    assert toolbar.actions()[1].text() == "Lasso"
    
    # Test button click
    with qtbot.waitSignal(toolbar.actionTriggered) as blocker:
        qtbot.mouseClick(toolbar.widgetForAction(toolbar.actions()[0]), Qt.LeftButton)
    
    # Verify signal was emitted with correct action
    assert blocker.args[0] == toolbar.actions()[0]
```

### 2. Interaction Tests
Tests for user interactions with the application.

#### Key Interactions to Test
- Mouse operations (click, double-click, drag, right-click)
- Keyboard operations (shortcuts, text input)
- Tool selection and usage
- Panel resizing and docking
- Dialog navigation and form submission

#### Example Interaction Test
```python
def test_lasso_selection(qtbot):
    """Test that lasso selection tool works correctly."""
    # Setup
    canvas = ImageCanvas()
    qtbot.addWidget(canvas)
    canvas.set_image(np.zeros((500, 500, 3), dtype=np.uint8))
    canvas.select_tool("lasso")
    
    # Simulate lasso selection
    start_pos = QPoint(100, 100)
    mid_pos1 = QPoint(200, 150)
    mid_pos2 = QPoint(150, 200)
    end_pos = QPoint(100, 100)  # Close the loop
    
    # Press, move, and release
    qtbot.mousePress(canvas, Qt.LeftButton, pos=start_pos)
    qtbot.mouseMove(canvas, mid_pos1)
    qtbot.mouseMove(canvas, mid_pos2)
    qtbot.mouseMove(canvas, end_pos)
    qtbot.mouseRelease(canvas, Qt.LeftButton, pos=end_pos)
    
    # Verify selection was created
    assert canvas.has_selection()
    assert len(canvas.selection_path.points) >= 4  # At least our 4 points
```

### 3. Visual Tests
Tests for the visual appearance and rendering of the UI.

#### Visual Elements to Test
- Theme application (light/dark mode)
- Icon rendering
- Layout and positioning
- Responsive design
- Custom styling

#### Example Visual Test
```python
def test_theme_switching(qtbot):
    """Test that theme switching changes the UI appearance."""
    # Setup
    main_window = MainWindow()
    qtbot.addWidget(main_window)
    main_window.show()
    
    # Capture initial state
    initial_screenshot = capture_screenshot(main_window)
    
    # Switch theme
    main_window.switch_theme("dark")
    qtbot.wait(100)  # Wait for theme to apply
    
    # Capture new state
    dark_screenshot = capture_screenshot(main_window)
    
    # Switch back to light
    main_window.switch_theme("light")
    qtbot.wait(100)  # Wait for theme to apply
    
    # Capture final state
    light_screenshot = capture_screenshot(main_window)
    
    # Verify screenshots are different for different themes
    assert not images_equal(initial_screenshot, dark_screenshot)
    assert images_equal(initial_screenshot, light_screenshot)  # Should be back to original
```

### 4. Workflow Tests
Tests for complete user workflows through the UI.

#### Key Workflows to Test
- Opening and saving images
- Applying filters and adjustments
- Creating and managing selections
- Working with layers
- Managing application preferences

#### Example Workflow Test
```python
def test_open_edit_save_workflow(qtbot, monkeypatch, tmpdir):
    """Test the basic workflow of opening, editing, and saving an image."""
    # Setup
    main_window = MainWindow()
    qtbot.addWidget(main_window)
    
    # Mock file dialogs
    test_image = os.path.join(os.path.dirname(__file__), "test_data", "sample.jpg")
    output_path = os.path.join(tmpdir, "output.jpg")
    
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args, **kwargs: (test_image, ""))
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args, **kwargs: (output_path, ""))
    
    # Execute workflow
    
    # 1. Open image
    qtbot.mouseClick(main_window.fileMenu, Qt.LeftButton)
    qtbot.wait(100)
    qtbot.mouseClick(main_window.openAction, Qt.LeftButton)
    qtbot.wait(500)  # Wait for image to load
    
    # 2. Apply an effect
    qtbot.mouseClick(main_window.filtersMenu, Qt.LeftButton)
    qtbot.wait(100)
    qtbot.mouseClick(main_window.gaussianBlurAction, Qt.LeftButton)
    
    # Wait for dialog to appear and set parameters
    qtbot.waitForWindowShown(main_window.activeWindow())
    dialog = main_window.activeWindow()
    slider = dialog.findChild(QSlider, "radiusSlider")
    qtbot.mouseClick(slider, Qt.LeftButton, pos=QPoint(slider.width() // 2, slider.height() // 2))
    qtbot.mouseClick(dialog.findChild(QPushButton, "okButton"), Qt.LeftButton)
    
    # 3. Save result
    qtbot.mouseClick(main_window.fileMenu, Qt.LeftButton)
    qtbot.wait(100)
    qtbot.mouseClick(main_window.saveAsAction, Qt.LeftButton)
    qtbot.wait(500)  # Wait for save to complete
    
    # Verify result
    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0
```

### 5. Stress Tests
Tests for UI behavior under high load or extreme conditions.

#### Stress Scenarios to Test
- Large images (8K+)
- Many layers (50+)
- Complex selections
- Rapid user interactions
- Low memory conditions

#### Example Stress Test
```python
@pytest.mark.slow
def test_large_image_handling(qtbot):
    """Test that the UI remains responsive with very large images."""
    # Setup
    main_window = MainWindow()
    qtbot.addWidget(main_window)
    
    # Create a large image (8K)
    large_image = np.zeros((7680, 4320, 3), dtype=np.uint8)
    
    # Start performance monitoring
    ui_response_times = []
    
    # Load the image
    with qtbot.waitSignal(main_window.image_loaded, timeout=30000):
        main_window.load_image_data(large_image)
    
    # Test UI responsiveness with various operations
    operations = [
        (main_window.zoom_in, "Zoom In"),
        (main_window.zoom_out, "Zoom Out"),
        (main_window.pan_left, "Pan Left"),
        (main_window.pan_right, "Pan Right")
    ]
    
    for operation, name in operations:
        start_time = time.time()
        operation()
        qtbot.wait(100)  # Wait for UI to update
        end_time = time.time()
        ui_response_times.append((name, end_time - start_time))
    
    # Verify that all operations complete within acceptable time
    for name, response_time in ui_response_times:
        assert response_time < 0.5, f"{name} took too long: {response_time}s"
```

## Test Data

### Test Images
- Standard test images in various formats (JPEG, PNG, TIFF)
- Images with different dimensions (small, medium, large)
- Images with different color spaces (RGB, CMYK, grayscale)
- Images with and without alpha channel
- RAW and HDR images (for format support testing)

### UI Test Fixtures
- Mock data for populating UI components
- Sample user preferences and settings
- Pre-defined selections and paths
- Sample layer stacks

## Testing Environment

### Local Development Testing
- Developer workstations
- Real mouse and keyboard
- Multiple monitors with different resolutions
- High DPI and standard DPI configurations

### CI/CD Testing
- Headless testing with Xvfb
- Virtual displays with fixed dimensions
- Simulated mouse and keyboard events
- Cross-platform testing (Windows, macOS, Linux)

## Testing Strategies

### 1. Component-Based Testing
Test individual UI components in isolation:

```python
# Example: Test color picker component
def test_color_picker(qtbot):
    """Test the color picker widget functionality."""
    picker = ColorPickerWidget()
    qtbot.addWidget(picker)
    
    # Test initial state
    assert picker.current_color == QColor(0, 0, 0)  # Default is black
    
    # Test color selection
    red_swatch = picker.findChild(QWidget, "redSwatch")
    qtbot.mouseClick(red_swatch, Qt.LeftButton)
    assert picker.current_color == QColor(255, 0, 0)
    
    # Test custom color dialog
    custom_button = picker.findChild(QPushButton, "customColorButton")
    
    # Mock color dialog
    with patch('PyQt6.QtWidgets.QColorDialog.getColor') as mock_get_color:
        mock_get_color.return_value = QColor(0, 255, 0)  # Green
        qtbot.mouseClick(custom_button, Qt.LeftButton)
        
    assert picker.current_color == QColor(0, 255, 0)
    
    # Test signal emission
    with qtbot.waitSignal(picker.colorChanged) as blocker:
        blue_swatch = picker.findChild(QWidget, "blueSwatch")
        qtbot.mouseClick(blue_swatch, Qt.LeftButton)
    
    assert blocker.args[0] == QColor(0, 0, 255)
```

### 2. Snapshot Testing
Compare rendered UI components against known good references:

```python
def test_toolbar_appearance(qtbot):
    """Test the visual appearance of the toolbar."""
    toolbar = Toolbar()
    qtbot.addWidget(toolbar)
    toolbar.show()
    
    # Capture the current appearance
    current_snapshot = capture_widget_snapshot(toolbar)
    
    # Compare with reference snapshot
    reference_path = os.path.join(REFERENCE_DIR, "toolbar_reference.png")
    if os.path.exists(reference_path):
        reference_snapshot = read_image(reference_path)
        similarity = compare_images(current_snapshot, reference_snapshot)
        assert similarity > 0.95, "Toolbar appearance has changed significantly"
    else:
        # If reference doesn't exist, save this as the reference
        save_image(current_snapshot, reference_path)
        pytest.skip("Reference image created - run test again for comparison")
```

### 3. Event Sequence Testing
Test complex interaction sequences:

```python
def test_selection_and_move(qtbot):
    """Test selection followed by move operations."""
    canvas = ImageCanvas()
    qtbot.addWidget(canvas)
    canvas.show()
    canvas.set_image(np.zeros((500, 500, 3), dtype=np.uint8))
    
    # Define a sequence of events
    events = [
        # Select rectangle tool
        lambda: canvas.set_tool("rectangle"),
        
        # Draw selection
        lambda: qtbot.mousePress(canvas, Qt.LeftButton, pos=QPoint(100, 100)),
        lambda: qtbot.mouseMove(canvas, QPoint(200, 200)),
        lambda: qtbot.mouseRelease(canvas, Qt.LeftButton, pos=QPoint(200, 200)),
        
        # Verify selection
        lambda: assert_true(canvas.has_selection()),
        
        # Switch to move tool
        lambda: canvas.set_tool("move"),
        
        # Move selection
        lambda: qtbot.mousePress(canvas, Qt.LeftButton, pos=QPoint(150, 150)),
        lambda: qtbot.mouseMove(canvas, QPoint(250, 250)),
        lambda: qtbot.mouseRelease(canvas, Qt.LeftButton, pos=QPoint(250, 250)),
        
        # Verify selection moved
        lambda: assert_equal(canvas.selection_rect.center(), QPoint(250, 250))
    ]
    
    # Execute sequence with small delays to simulate real usage
    for event in events:
        event()
        qtbot.wait(50)
```

### 4. Property-Based Testing
Generate random UI interactions to find edge cases:

```python
@pytest.mark.parametrize("n_interactions", [10, 50, 100])
def test_random_interactions(qtbot, n_interactions):
    """Test random sequences of UI interactions for robustness."""
    main_window = MainWindow()
    qtbot.addWidget(main_window)
    main_window.show()
    
    # Load a test image
    test_image = np.zeros((500, 500, 3), dtype=np.uint8)
    main_window.load_image_data(test_image)
    
    # Define possible actions
    actions = [
        lambda: main_window.zoom_in(),
        lambda: main_window.zoom_out(),
        lambda: main_window.pan_to(random.randint(0, 500), random.randint(0, 500)),
        lambda: main_window.select_tool(random.choice(["rectangle", "ellipse", "lasso", "wand"])),
        lambda: main_window.apply_filter(random.choice(["blur", "sharpen", "invert", "grayscale"])),
        # Add more possible actions
    ]
    
    # Perform random actions
    for _ in range(n_interactions):
        action = random.choice(actions)
        try:
            action()
            qtbot.wait(10)  # Small delay between actions
        except Exception as e:
            pytest.fail(f"Exception occurred during random interaction: {e}")
    
    # Verify application is still responsive
    assert main_window.isVisible()
    assert not main_window.isWindowModified()  # No unexpected state changes
```

## Implementation Plan

### Phase 1: Framework Setup (Week 1)
1. Configure pytest-qt with necessary plugins
2. Create UI test directory structure
3. Define test fixtures and helpers
4. Create utility functions for UI testing

### Phase 2: Component Test Implementation (Week 2)
1. Identify all UI components to test
2. Implement tests for main window components
3. Implement tests for dialogs and forms
4. Implement tests for custom widgets

### Phase 3: Interaction Test Implementation (Week 3)
1. Implement mouse interaction tests
2. Implement keyboard shortcut tests
3. Implement tool operation tests
4. Implement panel and docking tests

### Phase 4: Workflow Test Implementation (Week 4)
1. Identify key user workflows
2. Implement end-to-end workflow tests
3. Implement cross-component interaction tests
4. Implement error handling and recovery tests

### Phase 5: Visual and Stress Testing (Week 5)
1. Implement visual testing framework
2. Create visual reference images
3. Implement stress tests for extreme cases
4. Implement accessibility tests

## CI/CD Integration

### GitHub Actions Workflow
```yaml
name: UI Tests

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: [3.9, 3.10]

    steps:
    - uses: actions/checkout@v2
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Install Qt dependencies on Linux
      if: runner.os == 'Linux'
      run: |
        sudo apt-get update
        sudo apt-get install -y xvfb libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-xinerama0 libxcb-xkb1 libxkbcommon-x11-0
    
    - name: Run UI tests on Linux
      if: runner.os == 'Linux'
      run: |
        xvfb-run --auto-servernum pytest tests/ui
    
    - name: Run UI tests on Windows/macOS
      if: runner.os != 'Linux'
      run: |
        pytest tests/ui
```

## Reporting and Monitoring

### Test Report Generation
- Generate HTML reports with screenshots
- Track UI test coverage over time
- Document UI component test status

### Visual Regression Detection
- Compare screenshots with baselines
- Highlight visual differences
- Archive reference screenshots

### Performance Monitoring
- Track UI responsiveness over time
- Monitor rendering performance
- Identify slow UI operations

## Timeline
- Phase 1: 1 week
- Phase 2: 1 week
- Phase 3: 1 week
- Phase 4: 1 week
- Phase 5: 1 week
- Total: 5 weeks

## Quality Checklist
- [ ] UI test framework is configured and documented
- [ ] All critical UI components have tests
- [ ] All key user workflows are tested
- [ ] Visual tests verify UI appearance
- [ ] Stress tests verify UI under extreme conditions
- [ ] Tests run on all supported platforms
- [ ] CI/CD pipeline runs UI tests automatically
- [ ] UI test coverage meets quality targets
- [ ] Accessibility requirements are tested and met 