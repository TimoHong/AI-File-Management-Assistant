import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.core.classifier.image_classifier import ImageClassifier

def test_with_real_images():
    """Test the classifier with real images from a specified directory"""
    print("Testing Image Classifier with Real Images")
    
    # Get directory with real images from user
    image_dir = input("Enter the path to directory containing real images: ").strip()
    
    if not os.path.exists(image_dir):
        print(f"Directory not found: {image_dir}")
        return
    
    # Get list of image files
    image_files = []
    for file in os.listdir(image_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_files.append(os.path.join(image_dir, file))
    
    if not image_files:
        print("No image files found in the specified directory")
        return
    
    try:
        # Initialize classifier
        print("\nInitializing ImageClassifier...")
        classifier = ImageClassifier()
        
        # Analyze images
        print("\nAnalyzing images...")
        results = classifier.analyze_images(image_files)
        
        # Print individual results
        print("\nIndividual Image Analysis:")
        for result in results:
            print(f"\nFile: {result['filename']}")
            print(f"Caption: {result['caption']}")
        
        # Test grouping with different numbers of groups
        for num_groups in [2, 3, 4]:
            print(f"\nTesting grouping with {num_groups} groups:")
            grouped_images = classifier.group_images(results, num_groups=num_groups)
            
            for group_id, files in grouped_images.items():
                print(f"\nGroup {group_id}:")
                for file_path in files:
                    print(f"  - {os.path.basename(file_path)}")
                    
        print("\nTesting completed successfully!")
        
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        raise

if __name__ == "__main__":
    test_with_real_images()