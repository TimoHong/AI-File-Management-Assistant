from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from sklearn.cluster import KMeans
import logging
import os
from typing import List, Dict, Any
from torch.nn.functional import cosine_similarity
import numpy as np

class ImageClassifier:
    def __init__(self):
        try:
            # Initialize CLIP model
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.model.to(self.device)
            logging.info("CLIP model loaded successfully")
            
            # Define common categories for better grouping
            self.categories = [
                "landscape", "portrait", "document", "artwork",
                "building", "nature", "food", "animal",
                "vehicle", "indoor scene", "outdoor scene", "abstract",
                "group photo", "object", "text", "diagram"
            ]
            
        except Exception as e:
            logging.error(f"Error loading CLIP model: {str(e)}")
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

    def group_images(self, analysis_results: List[Dict[str, Any]], num_groups: int = 3) -> Dict[int, List[str]]:
        """Group images based on their features and similarities"""
        try:
            if len(analysis_results) < num_groups:
                num_groups = len(analysis_results)
            
            # Extract features and create similarity matrix
            features = np.stack([result['features'] for result in analysis_results])
            
            # Use K-means clustering with cosine similarity
            kmeans = KMeans(
                n_clusters=num_groups,
                n_init=10,
                random_state=42
            )
            
            # Normalize features before clustering
            normalized_features = features / np.linalg.norm(features, axis=1, keepdims=True)
            clusters = kmeans.fit_predict(normalized_features)
            
            # Organize results by group
            grouped_images = {i: [] for i in range(num_groups)}
            
            # Calculate central image for each cluster
            for cluster_id in range(num_groups):
                cluster_mask = clusters == cluster_id
                cluster_features = features[cluster_mask]
                cluster_paths = [result['path'] for i, result in enumerate(analysis_results) if clusters[i] == cluster_id]
                
                if len(cluster_paths) > 0:
                    # Calculate mean feature vector for the cluster
                    mean_feature = cluster_features.mean(axis=0)
                    
                    # Find images closest to cluster center
                    distances = [
                        np.linalg.norm(feature - mean_feature)
                        for feature in cluster_features
                    ]
                    
                    # Sort paths by distance to cluster center
                    sorted_paths = [x for _, x in sorted(zip(distances, cluster_paths))]
                    grouped_images[cluster_id] = sorted_paths
            
            return grouped_images
            
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