from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer
from PIL import Image
import torch
import os
from typing import List, Dict, Any
import logging

class ImageClassifier:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            # Initialize BLIP model for image captioning
            self.caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            
            # Initialize sentence transformer for text similarity
            self.text_model = SentenceTransformer("bert-base-nli-mean-tokens")
            
            logging.info("AI models loaded successfully")
        except Exception as e:
            logging.error(f"Error loading AI models: {str(e)}")
            raise

    def get_image_caption(self, image_path: str) -> str:
        """Generate caption for an image"""
        try:
            with Image.open(image_path).convert('RGB') as raw_image:
                inputs = self.caption_processor(raw_image, return_tensors="pt")
                out = self.caption_model.generate(**inputs)
                caption = self.caption_processor.decode(out[0], skip_special_tokens=True)
                return caption
        except Exception as e:
            logging.error(f"Error generating caption for {image_path}: {str(e)}")
            return ""

    def analyze_images(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """Analyze multiple images and return their information"""
        results = []
        
        for path in image_paths:
            try:
                if not os.path.exists(path):
                    continue
                    
                caption = self.get_image_caption(path)
                
                result = {
                    'path': path,
                    'caption': caption,
                    'filename': os.path.basename(path),
                    'type': 'image'
                }
                
                results.append(result)
                
            except Exception as e:
                logging.error(f"Error analyzing image {path}: {str(e)}")
                continue
                
        return results

    def group_images(self, analysis_results: List[Dict[str, Any]], num_groups: int = 3) -> Dict[int, List[str]]:
        """Group images based on caption similarity"""
        try:
            from sklearn.cluster import KMeans
            
            # Extract captions
            captions = [result['caption'] for result in analysis_results]
            
            # Get embeddings for captions
            embeddings = self.text_model.encode(captions)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=num_groups, random_state=42)
            clusters = kmeans.fit_predict(embeddings)
            
            # Organize results by group
            grouped_images = {i: [] for i in range(num_groups)}
            for idx, cluster in enumerate(clusters):
                grouped_images[cluster].append(analysis_results[idx]['path'])
                
            return grouped_images
            
        except Exception as e:
            logging.error(f"Error grouping images: {str(e)}")
            return {}