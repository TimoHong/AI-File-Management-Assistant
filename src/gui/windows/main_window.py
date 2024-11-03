from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QFileDialog, QTreeView, 
                            QListView, QStatusBar, QToolBar, QStyle, QSplitter,
                            QLineEdit, QProgressBar, QFrame, QMessageBox,
                            QButtonGroup, QRadioButton)
from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtGui import QIcon, QStandardItemModel, QStandardItem
from src.core.classifier.image_classifier import ImageClassifier
from src.core.classifier.document_classifier import DocumentClassifier
import os
import shutil
import logging

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.files_to_process = []
        self.current_progress = 0
        self.classifier = None
        self.doc_classifier = None
        self.setup_classifiers()

    def setup_classifiers(self):
        """Initialize the classifiers"""
        try:
            self.classifier = ImageClassifier()
            self.doc_classifier = DocumentClassifier()
            self.status_bar.showMessage('AI models loaded successfully')
        except Exception as e:
            self.status_bar.showMessage(f'Error loading AI models: {str(e)}')
            logging.error(f"Error loading classifiers: {str(e)}")

    def init_ui(self):
        # Set window properties
        self.setWindowTitle('AI File Management Assistant')
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QPushButton {
                background-color: #0078D4;
                color: white;
                border: none;
                padding: 5px 15px;
                border-radius: 4px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #106EBE;
            }
            QPushButton:disabled {
                background-color: #CCE4F6;
            }
            QTreeView, QListView {
                background-color: white;
                border: 1px solid #CCCCCC;
                border-radius: 4px;
                padding: 5px;
            }
            QStatusBar {
                background-color: #F5F5F5;
            }
            QToolBar {
                background-color: white;
                border-bottom: 1px solid #CCCCCC;
                padding: 5px;
            }
            QProgressBar {
                border: 1px solid #CCCCCC;
                border-radius: 3px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #0078D4;
                border-radius: 2px;
            }
            QRadioButton {
                padding: 5px;
            }
        """)

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Create toolbar
        self.create_toolbar()

        # Create mode selection
        mode_layout = QHBoxLayout()
        self.mode_group = QButtonGroup()
        
        self.image_mode = QRadioButton("Image Mode")
        self.doc_mode = QRadioButton("Document Mode")
        self.image_mode.setChecked(True)
        
        self.mode_group.addButton(self.image_mode)
        self.mode_group.addButton(self.doc_mode)
        
        mode_layout.addWidget(self.image_mode)
        mode_layout.addWidget(self.doc_mode)
        mode_layout.addStretch()
        
        main_layout.addLayout(mode_layout)

        # Create splitter for tree and list views
        splitter = QSplitter(Qt.Horizontal)
        
        # Left side - Directory Tree
        self.tree_view = QTreeView()
        self.tree_view.setHeaderHidden(True)
        self.tree_model = QStandardItemModel()
        self.tree_model.setHorizontalHeaderLabels(['Folders'])
        self.tree_view.setModel(self.tree_model)
        self.populate_tree()
        splitter.addWidget(self.tree_view)

        # Right side - File List and Controls
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Search bar
        search_layout = QHBoxLayout()
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search files...")
        self.search_bar.setStyleSheet("""
            QLineEdit {
                padding: 5px;
                border: 1px solid #CCCCCC;
                border-radius: 4px;
            }
        """)
        search_layout.addWidget(self.search_bar)
        right_layout.addLayout(search_layout)

        # File list
        self.file_list = QListView()
        self.file_model = QStandardItemModel()
        self.file_list.setModel(self.file_model)
        right_layout.addWidget(self.file_list)

        # Status and Progress Section
        status_frame = QFrame()
        status_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        status_layout = QVBoxLayout(status_frame)
        
        # File Counter
        self.file_counter_label = QLabel('Files: 0')
        status_layout.addWidget(self.file_counter_label)
        
        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat('%p% - %v/%m files')
        self.progress_bar.setVisible(False)
        status_layout.addWidget(self.progress_bar)
        
        # Current Operation Label
        self.operation_label = QLabel('')
        status_layout.addWidget(self.operation_label)
        
        right_layout.addWidget(status_frame)

        # Control buttons
        button_layout = QHBoxLayout()
        self.add_button = QPushButton('Add Files')
        self.organize_button = QPushButton('Organize')
        self.clear_button = QPushButton('Clear')
        
        button_layout.addWidget(self.add_button)
        button_layout.addWidget(self.organize_button)
        button_layout.addWidget(self.clear_button)
        button_layout.addStretch()
        
        right_layout.addLayout(button_layout)
        splitter.addWidget(right_widget)

        # Add splitter to main layout
        main_layout.addWidget(splitter)

        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage('Ready')

        # Connect signals
        self.add_button.clicked.connect(self.add_files)
        self.organize_button.clicked.connect(self.organize_files)
        self.clear_button.clicked.connect(self.clear_files)
        self.tree_view.clicked.connect(self.folder_selected)
        self.image_mode.toggled.connect(self.mode_changed)

    def create_toolbar(self):
        toolbar = QToolBar()
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)

        # Add actions
        refresh_action = toolbar.addAction('Refresh')
        refresh_action.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))

    def mode_changed(self):
        """Handle mode change between image and document"""
        if self.image_mode.isChecked():
            self.status_bar.showMessage('Switched to Image Mode')
        else:
            self.status_bar.showMessage('Switched to Document Mode')
        self.clear_files()

    def populate_tree(self):
        # Add some dummy items for now
        root_item = self.tree_model.invisibleRootItem()
        documents = QStandardItem('Documents')
        pictures = QStandardItem('Pictures')
        downloads = QStandardItem('Downloads')
        
        root_item.appendRow(documents)
        root_item.appendRow(pictures)
        root_item.appendRow(downloads)

    def add_files(self):
        """Add files using file dialog"""
        if self.image_mode.isChecked():
            file_filter = "Image Files (*.png *.jpg *.jpeg *.gif *.bmp);;All Files (*.*)"
        else:
            file_filter = "PDF Documents (*.pdf);;All Files (*.*)"

        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Files",
            "",
            file_filter
        )
        
        if files:
            self.files_to_process = files
            self.current_progress = 0
            self.progress_bar.setVisible(True)
            self.progress_bar.setMaximum(len(files))
            self.progress_bar.setValue(0)
            self.add_button.setEnabled(False)
            self.organize_button.setEnabled(False)
            
            # Start processing files
            self.process_next_file()

    def process_next_file(self):
        """Process files one by one"""
        if self.current_progress < len(self.files_to_process):
            file = self.files_to_process[self.current_progress]
            self.operation_label.setText(f'Processing: {os.path.basename(file)}')
            
            # Add file to list
            item = QStandardItem(file)
            self.file_model.appendRow(item)
            
            self.current_progress += 1
            self.progress_bar.setValue(self.current_progress)
            self.file_counter_label.setText(f'Files: {self.file_model.rowCount()}')
            
            # Schedule next file processing
            QTimer.singleShot(100, self.process_next_file)
        else:
            self.finish_processing()

    def finish_processing(self):
        """Finish file processing"""
        self.add_button.setEnabled(True)
        self.organize_button.setEnabled(True)
        self.operation_label.setText('Processing complete - Click Organize to group files')
        self.status_bar.showMessage(f'Added {len(self.files_to_process)} files')
        
        # Hide progress bar after a delay
        QTimer.singleShot(2000, lambda: self.progress_bar.setVisible(False))

    def organize_files(self):
        """Organize files based on selected mode"""
        if self.image_mode.isChecked():
            self.organize_images()
        else:
            self.organize_documents()

    def organize_images(self):
        """Organize images using AI classification"""
        if self.file_model.rowCount() == 0:
            QMessageBox.warning(self, "No Files", "Please add some images first.")
            return

        if not self.classifier:
            self.setup_classifiers()
            if not self.classifier:
                QMessageBox.critical(self, "Error", "Could not initialize AI model.")
                return

        try:
            # Collect all image paths from the model
            image_paths = []
            for row in range(self.file_model.rowCount()):
                file_path = self.file_model.item(row).text()
                if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                    image_paths.append(file_path)
                    print(f"Found image: {file_path}")

            if not image_paths:
                QMessageBox.warning(self, "No Images", "No valid image files found.")
                return

            # Show progress
            self.progress_bar.setVisible(True)
            self.progress_bar.setMaximum(len(image_paths))
            self.progress_bar.setValue(0)
            self.operation_label.setText('Analyzing images with AI...')
            self.add_button.setEnabled(False)
            self.organize_button.setEnabled(False)

            # Analyze images
            results = self.classifier.analyze_images(image_paths)
            
            # Group images
            grouped_images = self.classifier.group_images(results, num_groups=3)

            if grouped_images:
                # Create organized folders and move files
                base_dir = os.path.dirname(image_paths[0])
                organized_dir = os.path.join(base_dir, 'AI_Organized_Images')
                os.makedirs(organized_dir, exist_ok=True)

                for group_id, files in grouped_images.items():
                    # Get a representative caption for the group
                    group_images = [r for r in results if r['path'] in files]
                    group_caption = group_images[0]['caption'] if group_images else f'Group_{group_id}'
                    group_name = f"Group_{group_id}_{group_caption.replace(' ', '_')}"
                    
                    # Create group directory
                    group_dir = os.path.join(organized_dir, group_name)
                    os.makedirs(group_dir, exist_ok=True)

                    # Copy files to group directory
                    for file_path in files:
                        new_path = os.path.join(group_dir, os.path.basename(file_path))
                        shutil.copy2(file_path, new_path)

                # Show success message with details
                msg = (f"Successfully organized {len(image_paths)} images into {len(grouped_images)} groups.\n\n"
                      f"Location: {organized_dir}\n\n"
                      "Would you like to open the folder?")
                
                reply = QMessageBox.question(self, 'Organization Complete', msg,
                                           QMessageBox.Yes | QMessageBox.No)
                
                if reply == QMessageBox.Yes:
                    if os.name == 'nt':  # Windows
                        os.startfile(organized_dir)
                    else:  # macOS or Linux
                        import subprocess
                        subprocess.call(['open', organized_dir])

                # Clear the file list
                self.clear_files()

            else:
                QMessageBox.warning(self, "Error", "Could not group the images.")

        except Exception as e:
            import traceback
            print(f"\nError occurred: {str(e)}")
            print(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"Error organizing files: {str(e)}")

        finally:
            self.progress_bar.setVisible(False)
            self.add_button.setEnabled(True)
            self.organize_button.setEnabled(True)
            self.operation_label.setText('')

    def organize_documents(self):
        """Organize documents using AI classification"""
        if self.file_model.rowCount() == 0:
            QMessageBox.warning(self, "No Files", "Please add some documents first.")
            return

        try:
            # Initialize document classifier if needed
            if not self.doc_classifier:
                self.setup_classifiers()
                if not self.doc_classifier:
                    QMessageBox.critical(self, "Error", "Could not initialize document classifier.")
                    return

            # Collect all document paths from the model
            doc_paths = []
            for row in range(self.file_model.rowCount()):
                file_path = self.file_model.item(row).text()
                if file_path.lower().endswith('.pdf'):
                    doc_paths.append(file_path)
                    print(f"Found document: {file_path}")

            if not doc_paths:
                QMessageBox.warning(self, "No Documents", "No valid PDF documents found.")
                return

            # Show progress
            self.progress_bar.setVisible(True)
            self.progress_bar.setMaximum(len(doc_paths))
            self.progress_bar.setValue(0)
            self.operation_label.setText('Analyzing documents with AI...')
            self.add_button.setEnabled(False)
            self.organize_button.setEnabled(False)

            # Analyze documents
            results = self.doc_classifier.analyze_documents(doc_paths)
            
            # Group documents
            grouped_docs = self.doc_classifier.group_documents(results)

            if grouped_docs:
                # Create output directory
                base_dir = os.path.dirname(doc_paths[0])
                organized_dir = os.path.join(base_dir, 'AI_Organized_Documents')
                os.makedirs(organized_dir, exist_ok=True)

                # Move documents to their respective groups
                for category, files in grouped_docs.items():
                    category_dir = os.path.join(organized_dir, category)
                    os.makedirs(category_dir, exist_ok=True)

                    for file_path in files:
                        new_path = os.path.join(category_dir, os.path.basename(file_path))
                        shutil.copy2(file_path, new_path)

                # Show success message
                msg = (f"Successfully organized {len(doc_paths)} documents into {len(grouped_docs)} categories.\n\n"
                      f"Location: {organized_dir}\n\n"
                      "Would you like to open the folder?")
                
                reply = QMessageBox.question(self, 'Organization Complete', msg,
                                           QMessageBox.Yes | QMessageBox.No)
                
                if reply == QMessageBox.Yes:
                    if os.name == 'nt':  # Windows
                        os.startfile(organized_dir)
                    else:  # macOS or Linux
                        import subprocess
                        subprocess.call(['open', organized_dir])

                # Clear the file list
                self.clear_files()

            else:
                QMessageBox.warning(self, "Error", "Could not group the documents.")

        except Exception as e:
            import traceback
            print(f"\nError occurred: {str(e)}")
            print(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"Error organizing documents: {str(e)}")

        finally:
            self.progress_bar.setVisible(False)
            self.add_button.setEnabled(True)
            self.organize_button.setEnabled(True)
            self.operation_label.setText('')

    def folder_selected(self, index):
        """Handle folder selection"""
        item = self.tree_model.itemFromIndex(index)
        self.status_bar.showMessage(f'Selected folder: {item.text()}')

    def clear_files(self):
        """Clear the file list"""
        self.file_model.clear()
        self.file_counter_label.setText('Files: 0')
        self.progress_bar.setVisible(False)
        self.operation_label.clear()
        self.status_bar.showMessage('File list cleared')