import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
import torch
from transformers import AutoTokenizer, AutoModel
import pickle
from torch import nn
import fitz  # PyMuPDF
import logging
from typing import List, Dict, Any
import json
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import QApplication

class HierarchicalClassifier(nn.Module):
    def __init__(self, num_labels_level1=3, num_labels_level2=15, 
                 num_labels_level3=22, num_labels_level4=13):
        super().__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        hidden_size = self.bert.config.hidden_size
        
        self.level1_classifier = nn.Linear(hidden_size, num_labels_level1)
        self.level2_classifier = nn.Linear(hidden_size, num_labels_level2)
        self.level3_classifier = nn.Linear(hidden_size, num_labels_level3)
        self.level4_classifier = nn.Linear(hidden_size, num_labels_level4)
        
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        
        return (self.level1_classifier(pooled_output),
                self.level2_classifier(pooled_output),
                self.level3_classifier(pooled_output),
                self.level4_classifier(pooled_output))

class DocumentClassifier:
    def __init__(self, model_dir: str = None):
        if model_dir is None:
            # Use default path relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.join(current_dir, '..', '..', 'models', 'document_classifier_model')
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model configuration
        with open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
            model_config = json.load(f)
            
        # Load label encoders
        with open(os.path.join(model_dir, 'label_encoders.pkl'), 'rb') as f:
            self.label_encoders = pickle.load(f)
            
        # Initialize model
        self.model = HierarchicalClassifier(
            num_labels_level1=len(self.label_encoders['level1'].classes_),
            num_labels_level2=len(self.label_encoders['level2'].classes_),
            num_labels_level3=len(self.label_encoders['level3'].classes_),
            num_labels_level4=len(self.label_encoders['level4'].classes_)
        ).to(self.device)
        
        # Load model weights
        checkpoint = torch.load(os.path.join(model_dir, 'best_model.pt'), 
                              map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Initialize tokenizer
        tokenizer_path = os.path.join(model_dir, 'tokenizer')
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            doc = fitz.open(pdf_path)
            text = []
            for page in doc:
                text.append(page.get_text())
            doc.close()
            return " ".join(text).strip()
        except Exception as e:
            logging.error(f"Error reading PDF {pdf_path}: {str(e)}")
            return None

    def classify_document(self, text: str) -> List[str]:
        """Classify text using the hierarchical model"""
        if not text:
            return None
        
        try:
            # Tokenize input
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            # Prepare inputs
            model_inputs = {
                'input_ids': encoding['input_ids'].to(self.device),
                'attention_mask': encoding['attention_mask'].to(self.device)
            }
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**model_inputs)
            
            # Convert predictions to labels
            predictions = []
            confidence_scores = []
            for i, output in enumerate(outputs, 1):
                probs = torch.softmax(output, dim=1)
                confidence, pred_idx = probs.max(dim=1)
                pred_label = self.label_encoders[f'level{i}'].inverse_transform([pred_idx.item()])[0]
                predictions.append(pred_label)
                confidence_scores.append(confidence.item())
            
            return predictions, confidence_scores
            
        except Exception as e:
            logging.error(f"Error classifying document: {str(e)}")
            return None, None

    def analyze_documents(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Analyze multiple documents and return their information"""
        results = []
        
        for path in file_paths:
            try:
                if not os.path.exists(path):
                    continue
                    
                if path.lower().endswith('.pdf'):
                    # Extract text
                    text = self.extract_text_from_pdf(path)
                    if not text:
                        continue

                    # Truncate text if too long
                    if len(text) > 10000:
                        text = text[:10000]

                    # Get predictions
                    predictions, confidence_scores = self.classify_document(text)
                    if not predictions:
                        continue

                    result = {
                        'path': path,
                        'categories': predictions,
                        'confidence': confidence_scores,
                        'filename': os.path.basename(path),
                        'type': 'document'
                    }
                    results.append(result)
                
            except Exception as e:
                logging.error(f"Error analyzing document {path}: {str(e)}")
                continue
                
        return results

    def group_documents(self, analysis_results: List[Dict[str, Any]], 
                       confidence_threshold: float = 0.5,
                       num_groups: int = None) -> Dict[str, List[str]]:
        """Group documents based on their hierarchical classifications"""
        try:
            grouped_docs = {}
            
            for result in analysis_results:
                if result['categories'] and result['confidence']:
                    # Use the highest confidence category level
                    max_conf_idx = max(range(len(result['confidence'])), 
                                    key=lambda i: result['confidence'][i])
                    
                    if result['confidence'][max_conf_idx] >= confidence_threshold:
                        # If num_groups is specified, only use that many levels of categorization
                        if num_groups is not None:
                            max_conf_idx = min(num_groups - 1, max_conf_idx)
                        
                        category_path = ' → '.join(result['categories'][:max_conf_idx + 1])
                        
                        if category_path not in grouped_docs:
                            grouped_docs[category_path] = []
                            
                        grouped_docs[category_path].append(result['path'])
            
            # If num_groups is specified and we have more groups than requested,
            # merge the smallest groups
            if num_groups is not None and len(grouped_docs) > num_groups:
                sorted_groups = sorted(grouped_docs.items(), 
                                     key=lambda x: len(x[1]))
                while len(grouped_docs) > num_groups:
                    smallest_key, smallest_files = sorted_groups.pop(0)
                    second_smallest_key = sorted_groups[0][0]
                    
                    # Merge smallest into second smallest
                    grouped_docs[second_smallest_key].extend(smallest_files)
                    del grouped_docs[smallest_key]
                    
                    # Resort the list
                    sorted_groups = sorted(grouped_docs.items(), 
                                         key=lambda x: len(x[1]))
            
            return grouped_docs
            
        except Exception as e:
            logging.error(f"Error grouping documents: {str(e)}")
            return {}

class WorkerThread(QThread):
    finished = pyqtSignal(object)
    progress = pyqtSignal(str)

    def __init__(self, classifier, file_paths):
        super().__init__()
        self.classifier = classifier
        self.file_paths = file_paths

    def run(self):
        results = self.classifier.analyze_documents(self.file_paths)
        self.finished.emit(results)

class MainWindow:
    def process_documents(self, file_paths):
        self.worker = WorkerThread(self.classifier, file_paths)
        self.worker.finished.connect(self.on_processing_complete)
        self.worker.start()

    def on_processing_complete(self, results):
        try:
            if QThread.currentThread() is QApplication.instance().thread():
                self.update_gui_with_results(results)
            else:
                QMetaObject.invokeMethod(self, "update_gui_with_results",
                                       Qt.QueuedConnection,
                                       Q_ARG(object, results))
        except Exception as e:
            print(f"Error updating GUI: {str(e)}")

    def update_gui_with_results(self, results):
        pass