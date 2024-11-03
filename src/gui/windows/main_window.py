# src/gui/windows/main_window.py

from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QFileDialog, QTreeView, 
                            QListView, QStatusBar, QToolBar, QStyle, QSplitter,
                            QLineEdit, QProgressBar, QFrame, QMessageBox)
from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtGui import QIcon, QStandardItemModel, QStandardItem
from src.core.classifier.image_classifier import ImageClassifier
import os
import shutil

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.files_to_process = []
        self.current_progress = 0
        self.classifier = None
        self.setup_classifier()

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
        """)

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Create toolbar
        self.create_toolbar()

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

    def create_toolbar(self):
        toolbar = QToolBar()
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)

        # Add actions
        refresh_action = toolbar.addAction('Refresh')
        refresh_action.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))

    def setup_classifier(self):
        """Initialize the image classifier"""
        try:
            print("Initializing ImageClassifier...")  # Debug print
            self.classifier = ImageClassifier()
            print("ImageClassifier initialized successfully")  # Debug print
            self.status_bar.showMessage('AI model loaded successfully')
        except Exception as e:
            print(f"Error initializing classifier: {str(e)}")  # Debug print
            self.status_bar.showMessage(f'Error loading AI model: {str(e)}')

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
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Images",
            "",
            "Image Files (*.png *.jpg *.jpeg *.gif *.bmp);;All Files (*.*)"
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
        self.operation_label.setText('Processing complete - Click Organize to group images')
        self.status_bar.showMessage(f'Added {len(self.files_to_process)} files')
        
        # Hide progress bar after a delay
        QTimer.singleShot(2000, lambda: self.progress_bar.setVisible(False))

    def organize_files(self):
        """Organize files using AI classification"""
        print("\n=== Starting Image Organization ===")
        
        if self.file_model.rowCount() == 0:
            QMessageBox.warning(self, "No Files", "Please add some images first.")
            return

        if not self.classifier:
            print("Initializing classifier")
            self.setup_classifier()
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

            # Get the source directory from the first image
            source_dir = os.path.dirname(image_paths[0])
            print(f"\nSource directory: {source_dir}")

            # Create the output directory
            organized_dir = os.path.join(source_dir, 'AI_Organized_Images')
            print(f"Creating output directory: {organized_dir}")
            os.makedirs(organized_dir, exist_ok=True)

            # Show progress
            self.progress_bar.setVisible(True)
            self.progress_bar.setMaximum(len(image_paths))
            self.progress_bar.setValue(0)
            self.operation_label.setText('Analyzing images with AI...')
            self.add_button.setEnabled(False)
            self.organize_button.setEnabled(False)

            # Analyze images
            print("\nAnalyzing images...")
            results = []
            for idx, path in enumerate(image_paths):
                print(f"Processing image {idx + 1}/{len(image_paths)}: {os.path.basename(path)}")
                self.operation_label.setText(f'Analyzing image {idx + 1} of {len(image_paths)}...')
                result = self.classifier.analyze_images([path])[0]
                results.append(result)
                self.progress_bar.setValue(idx + 1)
                print(f"Caption generated: {result['caption']}")

            # Group images
            print("\nGrouping images...")
            self.operation_label.setText('Grouping similar images...')
            grouped_images = self.classifier.group_images(results, num_groups=3)

            # Create organized folders and move files
            if grouped_images:
                for group_id, files in grouped_images.items():
                    # Get a representative caption for the group
                    group_images = [r for r in results if r['path'] in files]
                    group_caption = group_images[0]['caption'] if group_images else f'Group_{group_id}'
                    group_name = f"Group_{group_id}_{group_caption.replace(' ', '_')}"
                    
                    # Create group directory
                    group_dir = os.path.join(organized_dir, group_name)
                    print(f"\nCreating group directory: {group_dir}")
                    os.makedirs(group_dir, exist_ok=True)

                    # Copy files to group directory
                    for file_path in files:
                        new_path = os.path.join(group_dir, os.path.basename(file_path))
                        print(f"Copying {os.path.basename(file_path)} to {group_name}")
                        shutil.copy2(file_path, new_path)

                print(f"\nOrganization complete! Files organized in: {organized_dir}")

                # Show success message with details
                msg = (f"Successfully organized {len(image_paths)} images into {len(grouped_images)} groups.\n\n"
                      f"Location: {organized_dir}\n\n"
                      "Would you like to open the folder?")
                
                reply = QMessageBox.question(self, 'Organization Complete', msg,
                                           QMessageBox.Yes | QMessageBox.No)
                
                if reply == QMessageBox.Yes:
                    # Open the folder in file explorer
                    if os.name == 'nt':  # Windows
                        os.startfile(organized_dir)
                    else:  # macOS or Linux
                        import subprocess
                        subprocess.call(['open', organized_dir])

            else:
                print("Error: Could not group the images")
                QMessageBox.warning(self, "Error", "Could not group the images.")

        except Exception as e:
            import traceback
            print(f"\nError occurred: {str(e)}")
            print(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"Error organizing files: {str(e)}")

        finally:
            # Reset UI state
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