import os
import pytesseract
from PIL import Image
import PyPDF2
import docx
import textract
from typing import Optional

class TextExtractor:
    """Extracts text from various file formats"""
    
    def __init__(self):
        # Configure pytesseract path if needed
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        pass

    def extract_text(self, file_path: str) -> Optional[str]:
        """
        Extract text from a file based on its type
        Returns: Extracted text or None if extraction fails
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_extension = os.path.splitext(file_path)[1].lower()

        try:
            if file_extension in ['.jpg', '.jpeg', '.png', '.bmp']:
                return self.extract_text_from_image(file_path)
            elif file_extension == '.pdf':
                return self.extract_text_from_pdf(file_path)
            elif file_extension in ['.doc', '.docx']:
                return self.extract_text_from_word(file_path)
            elif file_extension == '.txt':
                return self.extract_text_from_txt(file_path)
            else:
                return None
        except Exception as e:
            print(f"Error extracting text from {file_path}: {str(e)}")
            return None

    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from image using OCR"""
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            raise Exception(f"Error extracting text from image: {str(e)}")

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ''
                for page in reader.pages:
                    text += page.extract_text() + '\n'
                return text.strip()
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")

    def extract_text_from_word(self, doc_path: str) -> str:
        """Extract text from Word document"""
        try:
            if doc_path.endswith('.docx'):
                doc = docx.Document(doc_path)
                text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
                return text.strip()
            else:
                # For .doc files, use textract
                text = textract.process(doc_path).decode('utf-8')
                return text.strip()
        except Exception as e:
            raise Exception(f"Error extracting text from Word document: {str(e)}")

    def extract_text_from_txt(self, txt_path: str) -> str:
        """Extract text from plain text file"""
        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except Exception as e:
            raise Exception(f"Error extracting text from text file: {str(e)}")