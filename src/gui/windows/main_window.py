from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                           QPushButton, QFileDialog, QLabel, QListWidget,
                           QMessageBox, QStyle, QProgressBar)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        # Set window properties
        self.setWindowTitle('AI File Management Assistant')
        self.setMinimumSize(1000, 600)
        self.setAcceptDrops(True)  # Enable drag and drop

        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # Create header
        header_layout = QHBoxLayout()
        title_label = QLabel('AI File Management Assistant')
        title_label.setStyleSheet('''
            font-size: 24px;
            font-weight: bold;
            margin: 10px;
            color: #2c3e50;
        ''')
        header_layout.addWidget(title_label)
        main_layout.addLayout(header_layout)

        # Create content area
        content_layout = QHBoxLayout()

        # Left panel - File list
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # File list
        self.file_list = QListWidget()
        self.file_list.setStyleSheet('''
            QListWidget {
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                padding: 5px;
                background-color: #ffffff;
            }
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #ecf0f1;
            }
            QListWidget::item:selected {
                background-color: #3498db;
                color: white;
            }
        ''')
        left_layout.addWidget(QLabel('Selected Files:'))
        left_layout.addWidget(self.file_list)

        # Buttons
        button_layout = QHBoxLayout()
        
        # Add files button
        self.add_button = QPushButton('Add Files')
        self.add_button.setIcon(self.style().standardIcon(QStyle.SP_FileIcon))
        self.add_button.clicked.connect(self.add_files)
        self.add_button.setStyleSheet('''
            QPushButton {
                background-color: #2ecc71;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
        ''')
        button_layout.addWidget(self.add_button)

        # Clear button
        self.clear_button = QPushButton('Clear List')
        self.clear_button.setIcon(self.style().standardIcon(QStyle.SP_TrashIcon))
        self.clear_button.clicked.connect(self.clear_files)
        self.clear_button.setStyleSheet('''
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        ''')
        button_layout.addWidget(self.clear_button)

        left_layout.addLayout(button_layout)
        content_layout.addWidget(left_panel)

        # Right panel - Actions and status
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Organization options
        options_label = QLabel('Organization Options')
        options_label.setStyleSheet('font-weight: bold; font-size: 16px;')
        right_layout.addWidget(options_label)

        # Organize button
        self.organize_button = QPushButton('Organize Files')
        self.organize_button.setIcon(self.style().standardIcon(QStyle.SP_DialogApplyButton))
        self.organize_button.clicked.connect(self.organize_files)
        self.organize_button.setStyleSheet('''
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        ''')
        right_layout.addWidget(self.organize_button)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet('''
            QProgressBar {
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #3498db;
            }
        ''')
        right_layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel('Ready')
        self.status_label.setStyleSheet('color: #7f8c8d;')
        right_layout.addWidget(self.status_label)

        right_layout.addStretch()
        content_layout.addWidget(right_panel)

        main_layout.addLayout(content_layout)

    def add_files(self):
        """Handle file selection through dialog"""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Files",
            "",
            "All Files (*.*)"
        )
        
        if files:
            self.file_list.addItems(files)
            self.status_label.setText(f'Added {len(files)} files')

    def clear_files(self):
        """Clear the file list"""
        self.file_list.clear()
        self.status_label.setText('File list cleared')

    def organize_files(self):
        """Handle file organization"""
        if self.file_list.count() == 0:
            QMessageBox.warning(
                self,
                "No Files",
                "Please select files to organize first."
            )
            return

        # Simulate organization process
        total_files = self.file_list.count()
        self.progress_bar.setMaximum(total_files)
        
        # TODO: Implement actual file organization logic
        self.progress_bar.setValue(total_files)
        self.status_label.setText('Organization complete!')

    def dragEnterEvent(self, event):
        """Handle drag enter event"""
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        """Handle drop event"""
        files = [url.toLocalFile() for url in event.mimeData().urls()]
        self.file_list.addItems(files)
        self.status_label.setText(f'Added {len(files)} files')
    
    def organization_complete(self, results):
        """Handle organization completion"""
        self.toggle_ui_elements(True)
        self.progress_bar.setValue(self.progress_bar.maximum())

        # Prepare detailed results message
        success_count = len(results['success'])
        fail_count = len(results['failed'])
    
        message = f"Organization complete!\n\n"
    
        # Add category counts
        if results['categories']['Images']:
            image_count = sum(len(files) for files in results['categories']['Images'].values())
            message += f"Images organized: {image_count}\n"
    
        if results['categories']['Documents']:
            doc_count = sum(len(files) for files in results['categories']['Documents'].values())
            message += f"Documents organized: {doc_count}\n"
    
         # Add failure information if any
        if fail_count > 0:
            message += f"\nFailed to organize: {fail_count} files\n"
            message += "Failed files:\n"
            for file, error in results['failed']:
                message += f"- {os.path.basename(file)}: {error}\n"

        # Show results dialog
        QMessageBox.information(self, "Organization Complete", message)
    
        # Update status
        self.status_label.setText('Organization complete!')
        self.file_list.clear()