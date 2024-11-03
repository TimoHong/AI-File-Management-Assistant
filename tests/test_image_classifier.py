import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.core.classifier.image_classifier import ImageClassifier
from PIL import Image
import numpy as np

def create_test_images():
    """Create some test images if they don't exist"""
    test_dir = os.path.join(os.path.dirname(__file__), 'test_files')
    os.makedirs(test_dir, exist_ok=True)
    
    # Create different types of test images
    images_to_create = [
        ('test_red.jpg', (255, 0, 0)),      # Red image
        ('test_blue.jpg', (0, 0, 255)),     # Blue image
        ('test_green.jpg', (0, 255, 0)),    # Green image
    ]
    
    created_files = []
    for filename, color in images_to_create:
        file_path = os.path.join(test_dir, filename)
        if not os.path.exists(file_path):
            # Create a 100x100 image with the specified color
            img_array = np.full((100, 100, 3), color, dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(file_path)
            print(f"Created test image: {filename}")
        created_files.append(file_path)
    
    return created_files

def test_image_classifier():
    print("Starting Image Classifier Test")
    
    # Create test images
    test_files = create_test_images()
    
    try:
        # Initialize the classifier
        print("\nInitializing ImageClassifier...")
        classifier = ImageClassifier()
        print("ImageClassifier initialized successfully")
        
        # Test individual image caption generation
        print("\nTesting individual image captioning:")
        for image_path in test_files:
            print(f"\nProcessing {os.path.basename(image_path)}:")
            caption = classifier.get_image_caption(image_path)
            print(f"Generated caption: {caption}")
        
        # Test batch analysis
        print("\nTesting batch image analysis:")
        results = classifier.analyze_images(test_files)
        for result in results:
            print(f"\nFile: {result['filename']}")
            print(f"Caption: {result['caption']}")
            print(f"Type: {result['type']}")
        
        # Test image grouping
        print("\nTesting image grouping:")
        grouped_images = classifier.group_images(results, num_groups=2)
        for group_id, files in grouped_images.items():
            print(f"\nGroup {group_id}:")
            for file_path in files:
                print(f"  - {os.path.basename(file_path)}")
        
        print("\nAll tests completed successfully!")
        
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        raise

if __name__ == "__main__":
    test_image_classifier()