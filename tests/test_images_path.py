# tests/test_images_path.py
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from src.core.classifier.image_classifier import ImageClassifier

def test_specific_folder():
    """Test the classifier with images from our test_images folder"""
    # Specify the exact path to test_images
    image_dir = r"D:\EE4016_Group7_Project_Ai_file_management_assistant\AI-File-Management-Assistant\tests\test_files\test_images"
    
    print(f"Testing Image Classifier with images from: {image_dir}")
    
    if not os.path.exists(image_dir):
        print(f"Directory not found: {image_dir}")
        return
    
    # Get list of image files
    image_files = []
    for file in os.listdir(image_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            full_path = os.path.join(image_dir, file)
            image_files.append(full_path)
            print(f"Found image: {file}")
    
    if not image_files:
        print("No image files found in the specified directory")
        return
    
    print(f"\nFound {len(image_files)} images to analyze")
    
    try:
        # Initialize classifier
        print("\nInitializing ImageClassifier...")
        classifier = ImageClassifier()
        print("Classifier initialized successfully")
        
        # Analyze each image individually
        print("\nAnalyzing individual images:")
        for image_path in image_files:
            print(f"\nProcessing: {os.path.basename(image_path)}")
            caption = classifier.get_image_caption(image_path)
            print(f"Caption: {caption}")
        
        # Analyze all images as a batch
        print("\nPerforming batch analysis...")
        results = classifier.analyze_images(image_files)
        
        # Test grouping with different numbers of groups
        for num_groups in [2, 3]:
            print(f"\nGrouping images into {num_groups} groups:")
            grouped_images = classifier.group_images(results, num_groups=num_groups)
            
            for group_id, files in grouped_images.items():
                print(f"\nGroup {group_id}:")
                for file_path in files:
                    filename = os.path.basename(file_path)
                    # Find the caption for this file
                    caption = next((r['caption'] for r in results if r['path'] == file_path), 'No caption')
                    print(f"  - {filename}: {caption}")
                    
        print("\nTesting completed successfully!")
        
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    test_specific_folder()