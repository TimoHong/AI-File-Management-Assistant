import os
import shutil
from datetime import datetime
from typing import List, Dict

class FileOrganizer:
    def __init__(self):
        # Supported file extensions
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
        self.document_extensions = ['.pdf', '.doc', '.docx', '.txt', '.rtf']
        
        # Basic categories (placeholder until AI classification is implemented)
        self.image_categories = ['unsorted_images']
        self.document_categories = ['unsorted_documents']

    def is_supported_file(self, file_path: str) -> bool:
        """Check if file type is supported"""
        ext = os.path.splitext(file_path)[1].lower()
        return ext in self.image_extensions or ext in self.document_extensions

    def get_file_type(self, file_path: str) -> str:
        """Determine if file is image or document"""
        ext = os.path.splitext(file_path)[1].lower()
        if ext in self.image_extensions:
            return 'image'
        elif ext in self.document_extensions:
            return 'document'
        return 'unknown'

    def create_organization_directory(self, base_path: str) -> str:
        """Create directory structure for organized files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        org_dir = os.path.join(base_path, f"Organized_Files_{timestamp}")
        
        # Create main directories
        os.makedirs(org_dir, exist_ok=True)
        os.makedirs(os.path.join(org_dir, "Images"), exist_ok=True)
        os.makedirs(os.path.join(org_dir, "Documents"), exist_ok=True)
        
        # Create category subdirectories
        for category in self.image_categories:
            os.makedirs(os.path.join(org_dir, "Images", category), exist_ok=True)
        
        for category in self.document_categories:
            os.makedirs(os.path.join(org_dir, "Documents", category), exist_ok=True)
        
        return org_dir

    def organize_files(self, files: List[str], destination: str) -> Dict:
        """Organize files into basic categories"""
        results = {
            'success': [],
            'failed': [],
            'categories': {'Images': {}, 'Documents': {}}
        }

        # Create organization directory
        org_dir = self.create_organization_directory(destination)

        for file_path in files:
            try:
                if not os.path.exists(file_path):
                    results['failed'].append((file_path, "File not found"))
                    continue

                if not self.is_supported_file(file_path):
                    results['failed'].append((file_path, "Unsupported file type"))
                    continue

                file_type = self.get_file_type(file_path)
                
                # Place in appropriate directory
                if file_type == 'image':
                    category = 'unsorted_images'
                    base_dir = os.path.join(org_dir, "Images", category)
                elif file_type == 'document':
                    category = 'unsorted_documents'
                    base_dir = os.path.join(org_dir, "Documents", category)
                else:
                    continue

                # Handle file copy
                filename = os.path.basename(file_path)
                if os.path.exists(os.path.join(base_dir, filename)):
                    base_name, ext = os.path.splitext(filename)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{base_name}_{timestamp}{ext}"

                dest_path = os.path.join(base_dir, filename)
                shutil.copy2(file_path, dest_path)

                # Track results
                results['success'].append(dest_path)
                if file_type == 'image':
                    results['categories']['Images'].setdefault(category, []).append(dest_path)
                else:
                    results['categories']['Documents'].setdefault(category, []).append(dest_path)

            except Exception as e:
                results['failed'].append((file_path, str(e)))

        return results