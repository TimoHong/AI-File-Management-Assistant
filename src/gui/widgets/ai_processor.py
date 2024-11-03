# src/gui/widgets/ai_processor.py
from PyQt5.QtCore import QThread, pyqtSignal
from src.core.classifier.image_classifier import ImageClassifier

class AIProcessThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, image_paths, num_groups):
        super().__init__()
        self.image_paths = image_paths
        self.num_groups = num_groups
        self.classifier = None

    def run(self):
        try:
            # Initialize classifier in the thread
            self.classifier = ImageClassifier()
            
            # Analyze images
            total = len(self.image_paths)
            results = []
            
            for i, path in enumerate(self.image_paths):
                result = self.classifier.analyze_images([path])[0]
                results.append(result)
                self.progress.emit(int((i + 1) / total * 100))
            
            # Group images
            grouped_images = self.classifier.group_images(results, self.num_groups)
            self.finished.emit(grouped_images)
            
        except Exception as e:
            self.error.emit(str(e))