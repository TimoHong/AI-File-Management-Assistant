from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
from sklearn.cluster import KMeans
import logging
import os
from typing import List, Dict, Any
from torch.nn.functional import cosine_similarity
import numpy as np
import spacy
from collections import Counter
import nltk
from typing import Set

# Download required NLTK data
nltk.download('averaged_perceptron_tagger')
nltk.download('words')

class ImageClassifier:
    def __init__(self):
        try:
            # Initialize CLIP model
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.model.to(self.device)
            logging.info("CLIP model loaded successfully")
            
            # Add BLIP initialization
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_model.to(self.device)
            
            # Initialize spaCy
            self.nlp = spacy.load("en_core_web_sm")
            
            # Define common categories for better grouping
            self.categories = [
                "landscape", "portrait", "document", "artwork",
                "building", "nature", "food", "animal",
                "vehicle", "indoor scene", "outdoor scene", "abstract",
                "group photo", "object", "text", "diagram"
            ]
            
        except Exception as e:
            logging.error(f"Error loading models: {str(e)}")
            raise

    def get_image_features(self, image_path: str) -> torch.Tensor:
        """Extract image features using CLIP"""
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            return image_features.cpu()
            
        except Exception as e:
            logging.error(f"Error processing image {image_path}: {str(e)}")
            return None

    def get_text_features(self, texts: List[str]) -> torch.Tensor:
        """Get text features using CLIP"""
        try:
            inputs = self.processor(text=texts, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                
            # Normalize features
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features.cpu()
            
        except Exception as e:
            logging.error(f"Error processing text: {str(e)}")
            return None

    def get_image_caption(self, image_path: str) -> str:
        """Get the best matching category for an image"""
        try:
            image_features = self.get_image_features(image_path)
            text_features = self.get_text_features(self.categories)
            
            if image_features is None or text_features is None:
                return ""
            
            # Calculate similarity scores
            similarity = cosine_similarity(image_features, text_features)
            best_match_idx = similarity.argmax().item()
            
            return self.categories[best_match_idx]
            
        except Exception as e:
            logging.error(f"Error getting caption for {image_path}: {str(e)}")
            return ""

    def analyze_images(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """Analyze multiple images and return their information"""
        results = []
        all_features = []
        
        for path in image_paths:
            try:
                if not os.path.exists(path):
                    continue
                
                # Get image features
                features = self.get_image_features(path)
                if features is None:
                    continue
                
                # Get category label
                caption = self.get_image_caption(path)
                
                result = {
                    'path': path,
                    'caption': caption,
                    'features': features.squeeze().numpy(),
                    'filename': os.path.basename(path),
                    'type': 'image'
                }
                results.append(result)
                all_features.append(features.squeeze().numpy())
                
            except Exception as e:
                logging.error(f"Error analyzing image {path}: {str(e)}")
                continue
        
        # Update results with similarity scores
        if len(results) > 1:
            similarity_matrix = self.calculate_similarity_matrix(all_features)
            for i, result in enumerate(results):
                result['similarity_scores'] = similarity_matrix[i]
                
        return results

    def calculate_similarity_matrix(self, features: List[np.ndarray]) -> np.ndarray:
        """Calculate similarity matrix between all images"""
        features = np.stack(features)
        similarity_matrix = np.zeros((len(features), len(features)))
        
        for i in range(len(features)):
            for j in range(len(features)):
                similarity_matrix[i][j] = cosine_similarity(
                    torch.tensor(features[i]).unsqueeze(0),
                    torch.tensor(features[j]).unsqueeze(0)
                )
                
        return similarity_matrix

    def group_images(self, analysis_results: List[Dict[str, Any]], num_groups: int = 3) -> Dict[str, List[str]]:
        """Group images and assign smart names to groups"""
        try:
            # Ensure we have enough images for the requested number of groups
            total_images = len(analysis_results)
            if total_images < num_groups:
                num_groups = max(1, total_images)
            
            # Extract and normalize features
            features = np.stack([result['features'] for result in analysis_results])
            normalized_features = features / np.linalg.norm(features, axis=1, keepdims=True)
            
            # Perform clustering with exact number of groups
            kmeans = KMeans(n_clusters=num_groups, n_init=10, random_state=42)
            clusters = kmeans.fit_predict(normalized_features)
            
            # Initialize groups
            grouped_images = {f"Group_{i}": [] for i in range(num_groups)}
            
            # Organize images into groups
            for i, result in enumerate(analysis_results):
                cluster_id = clusters[i]
                grouped_images[f"Group_{cluster_id}"].append(result['path'])
            
            # Generate smart names for groups
            smart_grouped_images = {}
            existing_names = set()
            
            for group_name, image_paths in grouped_images.items():
                if image_paths:  # Only process non-empty groups
                    smart_name = self.get_smart_group_name(image_paths, existing_names)
                    smart_grouped_images[smart_name] = image_paths
                    existing_names.add(smart_name)
            
            logging.info(f"Created {len(smart_grouped_images)} groups")  # Debug log
            return smart_grouped_images
            
        except Exception as e:
            logging.error(f"Error grouping images: {str(e)}")
            return {}

    def get_group_description(self, group_images: List[str]) -> str:
        """Generate a description for a group of images"""
        try:
            # Get categories for all images in the group
            categories = [self.get_image_caption(img_path) for img_path in group_images]
            
            # Count category occurrences
            category_counts = {}
            for cat in categories:
                category_counts[cat] = category_counts.get(cat, 0) + 1
                
            # Get most common category
            most_common = max(category_counts.items(), key=lambda x: x[1])
            return most_common[0]
            
        except Exception as e:
            logging.error(f"Error getting group description: {str(e)}")
            return "Mixed"

    def is_valid_sentence(self, caption: str) -> bool:
        """Check if the caption has at least one noun and one verb and is long enough."""
        doc = self.nlp(caption)
        has_noun = any(token.pos_ == 'NOUN' for token in doc)
        has_verb = any(token.pos_ == 'VERB' for token in doc)
        return len(doc) > 3 and has_noun and has_verb

    def generate_blip_captions(self, image_paths: List[str], max_images: int = 10) -> List[str]:
        """Generate captions using BLIP model for the first few images."""
        captions = []
        limited_paths = image_paths[:max_images]

        for image_path in limited_paths:
            try:
                image = Image.open(image_path)
                inputs = self.blip_processor(images=image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                output = self.blip_model.generate(**inputs)
                caption = self.blip_processor.decode(output[0], skip_special_tokens=True)
                captions.append(caption)
            except Exception as e:
                logging.error(f"Error generating caption for {image_path}: {str(e)}")

        return captions

    def generate_group_name(self, captions: List[str]) -> str:
        """Generate a suitable group name based on captions."""
        all_nouns = []
        all_verbs = []

        for caption in captions:
            doc = self.nlp(caption)
            nouns = [token.lemma_ for token in doc if token.pos_ == 'NOUN']
            verbs = [token.lemma_ for token in doc if token.pos_ == 'VERB']
            all_nouns.extend(nouns)
            all_verbs.extend(verbs)

        common_noun = Counter(all_nouns).most_common(1)
        common_verb = Counter(all_verbs).most_common(1)
        
        new_name = f"{common_noun[0][0] if common_noun else 'Group'}_{common_verb[0][0] if common_verb else 'Activity'}"
        return new_name

    def ensure_unique_name(self, new_name: str, existing_names: Set[str]) -> str:
        """Ensure the new name is unique by adding a number if necessary."""
        count = 1
        base_name = new_name
        while new_name in existing_names:
            new_name = f"{base_name}_{count}"
            count += 1
        return new_name

    def get_smart_group_name(self, group_images: List[str], existing_names: Set[str]) -> str:
        """Generate a smart name for a group of images using BLIP and NLP."""
        try:
            # Generate captions using BLIP
            captions = self.generate_blip_captions(group_images)
            
            # Filter valid captions
            valid_captions = [caption for caption in captions if self.is_valid_sentence(caption)]
            
            if valid_captions:
                # Generate name based on captions
                new_name = self.generate_group_name(valid_captions)
                # Ensure unique name
                return self.ensure_unique_name(new_name, existing_names)
            else:
                # Fallback to original method if no valid captions
                return self.get_group_description(group_images)
                
        except Exception as e:
            logging.error(f"Error generating smart group name: {str(e)}")
            return self.get_group_description(group_images)