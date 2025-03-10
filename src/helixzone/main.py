"""Main entry point for HelixZone application."""

import cv2
import numpy as np
import argparse
from numpy.typing import NDArray
from helixzone.core.gpu import GPUManager, GPUBackend
from pathlib import Path
from typing import Optional, Dict
import time

def save_image(path: Path, array: NDArray[np.uint8]) -> None:
    """Save a numpy array as an image using OpenCV."""
    success = cv2.imwrite(str(path), array)
    if not success:
        raise RuntimeError(f"Failed to save image to {path}")

def create_sample_image(path: Path) -> None:
    """Create a sample image with geometric shapes."""
    print("Creating a sample test image...")
    img = np.zeros((512, 512), dtype=np.uint8)
    # Add some shapes
    cv2.circle(img, (256, 256), 100, (255,), 2)
    cv2.rectangle(img, (100, 100), (400, 400), (255,), 2)
    cv2.line(img, (50, 50), (450, 450), (255,), 2)
    # Save the image
    save_image(path, img)
    print(f"Created sample image at {path}")

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="HelixZone Edge Detection")
    parser.add_argument(
        "--image", "-i",
        type=str,
        help="Path to input image. If not provided, a sample image will be created."
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output directory for results. Default: test_images/output"
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["cuda", "opencl", "cpu"],
        help="GPU backend to use. Default: auto-detect"
    )
    parser.add_argument(
        "--bilateral-diameter", "-d",
        type=int,
        default=5,
        help="Bilateral filter diameter. Default: 5"
    )
    parser.add_argument(
        "--sigma-color",
        type=float,
        default=50.0,
        help="Bilateral filter sigma color. Default: 50.0"
    )
    parser.add_argument(
        "--sigma-space",
        type=float,
        default=50.0,
        help="Bilateral filter sigma space. Default: 50.0"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Edge detection threshold. Default: 0.1"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark mode with multiple iterations"
    )
    return parser.parse_args()

def process_image(gpu_manager: GPUManager, image: NDArray[np.uint8], args: argparse.Namespace) -> Dict[str, NDArray]:
    """Process a single image."""
    start_time = time.perf_counter()
    result = gpu_manager.process_edges(image, {
        'd': args.bilateral_diameter,
        'sigmaColor': args.sigma_color,
        'sigmaSpace': args.sigma_space,
        'threshold': args.threshold
    })
    end_time = time.perf_counter()
    print(f"Processing time: {(end_time - start_time) * 1000:.2f}ms")
    return result

def main() -> None:
    """Run the main application."""
    args = parse_args()
    
    # Initialize GPU manager
    gpu_manager = GPUManager()
    if args.backend:
        # Override auto-detection if backend specified
        gpu_manager._backend = args.backend  # type: ignore
    gpu_manager.initialize()
    
    # Print which backend we're using
    print(f"Using GPU backend: {gpu_manager.get_backend()}")
    
    # Setup paths
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            raise RuntimeError(f"Input image not found: {image_path}")
    else:
        # Use default sample image
        test_dir = Path(__file__).parent.parent.parent / "test_images"
        test_dir.mkdir(exist_ok=True)
        image_path = test_dir / "sample.jpg"
        if not image_path.exists():
            create_sample_image(image_path)
    
    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = image_path.parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Load the image
    print(f"Loading image from {image_path}")
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Failed to load image from {image_path}")
    
    # Ensure correct type
    image_uint8: NDArray[np.uint8] = np.asarray(image, dtype=np.uint8)
    
    # Process edges
    if args.benchmark:
        print("\nRunning benchmark mode...")
        iterations = 5
        times = []
        for i in range(iterations):
            print(f"\nIteration {i+1}/{iterations}")
            start_time = time.perf_counter()
            result = process_image(gpu_manager, image_uint8, args)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        # Print benchmark results
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        std_time = np.std(times)
        print("\nBenchmark Results:")
        print("-" * 50)
        print(f"Average time: {avg_time*1000:.2f}ms")
        print(f"Min time: {min_time*1000:.2f}ms")
        print(f"Max time: {max_time*1000:.2f}ms")
        print(f"Std dev: {std_time*1000:.2f}ms")
        print("-" * 50)
    else:
        print("Processing edges...")
        result = process_image(gpu_manager, image_uint8, args)
    
    print("Saving results...")
    save_image(output_dir / "edge_map.png", result['edge_map'])
    save_image(output_dir / "edge_strength.png", result['edge_strength'])
    
    # Visualize gradient direction
    gradient = result['edge_gradient']
    gradient_vis: NDArray[np.uint8] = np.asarray(
        (gradient + np.pi) * 255 / (2 * np.pi),
        dtype=np.uint8
    )
    save_image(output_dir / "edge_gradient.png", gradient_vis)
    
    print(f"Results saved to {output_dir}")
    print("Done!")

if __name__ == "__main__":
    main() 