import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from PyPDF2 import PdfReader
import pickle
from typing import List, Dict, Tuple, Any
import logging

class DocumentClassifier:
    # Main arXiv categories
    ARXIV_CATEGORIES = {
        'cs': 'Computer Science',
        'physics': 'Physics',
        'math': 'Mathematics',
        'q-bio': 'Quantitative Biology',
        'q-fin': 'Quantitative Finance'
    }

    def __init__(self, model_dir: str = None):
        if model_dir is None:
            # Use default path relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.join(current_dir, '..', '..', 'models', 'arxiv_classifier_model')

        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = BertForSequenceClassification.from_pretrained(
                model_dir,
                num_labels=len(self.ARXIV_CATEGORIES),
                ignore_mismatched_sizes=True
            )
            self.tokenizer = BertTokenizer.from_pretrained(model_dir)
            self.model.to(self.device)
            
            # Load label encoder
            with open(os.path.join(model_dir, 'label_encoder.pkl'), 'rb') as f:
                self.label_encoder = pickle.load(f)
                
            logging.info("Document classifier initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing document classifier: {str(e)}")
            raise

    def extract_text_from_pdf(self, pdf_path: str, max_chars: int = 10000) -> str:
        """Extract text from a PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                    if len(text) > max_chars:
                        return text[:max_chars]
                return text
        except Exception as e:
            logging.error(f"Error reading PDF {pdf_path}: {str(e)}")
            return ""

    def predict_categories(self, text: str, max_length: int = 128) -> List[Tuple[str, float]]:
        """Predict top 2 categories for a given text"""
        chunks = [text[i:i + max_length * 4] for i in range(0, len(text), max_length * 4)]
        all_probabilities = []

        for chunk in chunks[:5]:  # Process first 5 chunks only
            inputs = self.tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            self.model.eval()
            with torch.no_grad():
                outputs = self.model(**inputs)

            probabilities = torch.softmax(outputs.logits, dim=1)
            all_probabilities.append(probabilities)

        if all_probabilities:
            avg_probabilities = torch.mean(torch.cat(all_probabilities, dim=0), dim=0)
            top_2_values, top_2_indices = torch.topk(avg_probabilities, k=2)

            result = []
            for prob, idx in zip(top_2_values.cpu().numpy(), top_2_indices.cpu().numpy()):
                category = self.label_encoder.inverse_transform([idx])[0]
                result.append((category, float(prob)))

            return result

        return []

    def analyze_documents(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Analyze multiple documents and return their information"""
        results = []
        
        for path in file_paths:
            try:
                if not os.path.exists(path):
                    continue
                    
                if path.lower().endswith('.pdf'):
                    text = self.extract_text_from_pdf(path)
                    if text.strip():
                        categories = self.predict_categories(text)
                        result = {
                            'path': path,
                            'categories': categories,
                            'filename': os.path.basename(path),
                            'type': 'document'
                        }
                        results.append(result)
                
            except Exception as e:
                logging.error(f"Error analyzing document {path}: {str(e)}")
                continue
                
        return results

    def group_documents(self, analysis_results: List[Dict[str, Any]], num_groups: int = None) -> Dict[str, List[str]]:
        """Group documents based on their categories"""
        try:
            # Create groups based on primary categories
            grouped_docs = {}
            
            for result in analysis_results:
                if result['categories']:
                    primary_category = result['categories'][0][0]  # Get first category
                    category_name = f"{primary_category} - {self.ARXIV_CATEGORIES.get(primary_category, 'Other')}"
                    
                    if category_name not in grouped_docs:
                        grouped_docs[category_name] = []
                        
                    grouped_docs[category_name].append(result['path'])
            
            return grouped_docs
            
        except Exception as e:
            logging.error(f"Error grouping documents: {str(e)}")
            return {}