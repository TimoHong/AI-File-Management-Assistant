from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                           QPushButton, QFileDialog, QLabel, 
                           QListWidget, QMessageBox)
from PyQt5.QtCore import Qt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        # Set window properties
        self.setWindowTitle('AI File Management Assistant')
        self.setGeometry(100, 100, 800, 600)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Add title label
        title_label = QLabel('AI File Management Assistant')
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet('font-size: 20px; font-weight: bold; margin: 10px;')
        layout.addWidget(title_label)

        # Add file selection button
        self.select_button = QPushButton('Select Files')
        self.select_button.clicked.connect(self.select_files)
        layout.addWidget(self.select_button)

        # Add file list widget
        self.file_list = QListWidget()
        layout.addWidget(self.file_list)

        # Add organize button
        self.organize_button = QPushButton('Organize Files')
        self.organize_button.clicked.connect(self.organize_files)
        layout.addWidget(self.organize_button)

    def select_files(self):
        """Handle file selection"""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Files",
            "",
            "All Files (*.*)"
        )
        
        if files:
            self.file_list.clear()
            self.file_list.addItems(files)

    def organize_files(self):
        """Handle file organization"""
        if self.file_list.count() == 0:
            QMessageBox.warning(
                self,
                "No Files",
                "Please select files to organize first."
            )
            return

        # Get all files from the list
        files = [self.file_list.item(i).text() 
                for i in range(self.file_list.count())]
        
        # TODO: Implement actual file organization logic
        QMessageBox.information(
            self,
            "Not Implemented",
            f"File organization will be implemented soon.\nSelected files: {len(files)}"
        )