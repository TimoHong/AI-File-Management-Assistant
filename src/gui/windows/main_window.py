import os
import sys
from concurrent.futures import ThreadPoolExecutor

# Add the project root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Use relative imports
from src.core.classifier.image_classifier import ImageClassifier
from src.core.classifier.document_classifier import DocumentClassifier


from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QFileDialog, QTreeView, 
                            QListView, QStatusBar, QToolBar, QStyle, QSplitter,
                            QLineEdit, QProgressBar, QFrame, QMessageBox,
                            QButtonGroup, QRadioButton, QSlider)
from PyQt5.QtCore import Qt, QSize, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QIcon, QStandardItemModel, QStandardItem

class WorkerSignals(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    status = pyqtSignal(str)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.files_to_process = []
        self.current_progress = 0
        self.classifier = None
        self.doc_classifier = None
        self.thread_pool = ThreadPoolExecutor(max_workers=1)
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
            QSlider {
                height: 20px;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #CCCCCC;
                margin: 0px;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #0078D4;
                width: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }
        """)

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # Create toolbar
        self.create_toolbar()

        # Mode selection with better spacing and alignment
        mode_frame = QFrame()
        mode_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 8px;
                padding: 15px;
            }
        """)
        mode_layout = QHBoxLayout(mode_frame)
        
        # Mode selection group
        mode_group_layout = QHBoxLayout()
        self.mode_group = QButtonGroup()
        
        self.image_mode = QRadioButton("Image Mode")
        self.doc_mode = QRadioButton("Document Mode")
        self.image_mode.setChecked(True)
        
        self.mode_group.addButton(self.image_mode)
        self.mode_group.addButton(self.doc_mode)
        
        mode_group_layout.addWidget(self.image_mode)
        mode_group_layout.addWidget(self.doc_mode)
        mode_layout.addLayout(mode_group_layout)
        
        # Slider group with better spacing
        slider_layout = QHBoxLayout()
        self.groups_label = QLabel("Number of Groups:")
        self.groups_slider = QSlider(Qt.Horizontal)
        self.groups_slider.setMinimum(2)
        self.groups_slider.setMaximum(10)
        self.groups_slider.setValue(3)
        self.groups_slider.setTickPosition(QSlider.TicksBelow)
        self.groups_slider.setTickInterval(1)
        self.groups_value_label = QLabel("3")
        
        slider_layout.addWidget(self.groups_label)
        slider_layout.addWidget(self.groups_slider)
        slider_layout.addWidget(self.groups_value_label)
        mode_layout.addLayout(slider_layout)
        mode_layout.addStretch()
        
        main_layout.addWidget(mode_frame)

        # Main content area
        content_frame = QFrame()
        content_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 8px;
                padding: 15px;
            }
        """)
        content_layout = QVBoxLayout(content_frame)
        
        # Search bar with icon
        search_layout = QHBoxLayout()
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search files...")
        self.search_bar.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                padding-left: 15px;
                border: 1px solid #CCCCCC;
                border-radius: 20px;
                font-size: 14px;
            }
        """)
        search_layout.addWidget(self.search_bar)
        content_layout.addLayout(search_layout)

        # File list
        self.file_list = QListView()
        self.file_list.setStyleSheet("""
            QListView {
                border: 1px solid #CCCCCC;
                border-radius: 8px;
                padding: 5px;
                background-color: #FFFFFF;
            }
            QListView::item {
                padding: 8px;
                border-bottom: 1px solid #EEEEEE;
            }
            QListView::item:selected {
                background-color: #E3F2FD;
                color: #0078D4;
            }
        """)
        self.file_model = QStandardItemModel()
        self.file_list.setModel(self.file_model)
        content_layout.addWidget(self.file_list)

        # Status frame
        status_frame = QFrame()
        status_frame.setStyleSheet("""
            QFrame {
                background-color: #F8F9FA;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        status_layout = QVBoxLayout(status_frame)
        
        self.file_counter_label = QLabel('Files: 0')
        self.file_counter_label.setStyleSheet("font-weight: bold;")
        status_layout.addWidget(self.file_counter_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                border-radius: 10px;
                text-align: center;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #0078D4;
                border-radius: 10px;
            }
        """)
        self.progress_bar.setVisible(False)
        status_layout.addWidget(self.progress_bar)
        
        self.operation_label = QLabel('')
        status_layout.addWidget(self.operation_label)
        
        content_layout.addWidget(status_frame)

        # Button layout with modern styling
        button_layout = QHBoxLayout()
        self.add_button = QPushButton('Add Files')
        self.organize_button = QPushButton('Organize')
        self.clear_button = QPushButton('Clear')
        
        for btn in [self.add_button, self.organize_button, self.clear_button]:
            btn.setMinimumHeight(40)
            btn.setCursor(Qt.PointingHandCursor)
        
        self.organize_button.setStyleSheet("""
            QPushButton {
                background-color: #28A745;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        
        button_layout.addWidget(self.add_button)
        button_layout.addWidget(self.organize_button)
        button_layout.addWidget(self.clear_button)
        button_layout.addStretch()
        
        content_layout.addLayout(button_layout)
        main_layout.addWidget(content_frame)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage('Ready')

        # Connect signals
        self.add_button.clicked.connect(self.add_files)
        self.organize_button.clicked.connect(self.organize_files)
        self.clear_button.clicked.connect(self.clear_files)
        self.image_mode.toggled.connect(self.mode_changed)
        self.search_bar.textChanged.connect(self.filter_files)

    def update_groups_value(self, value):
        """Update the groups value label"""
        self.groups_value_label.setText(str(value))

    def create_toolbar(self):
        toolbar = QToolBar()
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)

        # Add actions
        refresh_action = toolbar.addAction('Refresh')
        refresh_action.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        refresh_action.triggered.connect(self.refresh_view)

    def refresh_view(self):
        """Refresh the current view"""
        self.status_bar.showMessage('Refreshing...')
        self.populate_tree()
        self.status_bar.showMessage('View refreshed')

    def filter_files(self, text):
        """Filter files in the list based on search text"""
        for row in range(self.file_model.rowCount()):
            item = self.file_model.item(row)
            file_name = os.path.basename(item.text()).lower()
            item.setHidden(text.lower() not in file_name)

    def mode_changed(self):
        """Handle mode change between image and document"""
        if self.image_mode.isChecked():
            self.status_bar.showMessage('Switched to Image Mode')
            self.groups_slider.setEnabled(True)
            self.groups_label.setEnabled(True)
            self.groups_value_label.setEnabled(True)
        else:
            self.status_bar.showMessage('Switched to Document Mode')
            self.groups_slider.setEnabled(False)
            self.groups_label.setEnabled(False)
            self.groups_value_label.setEnabled(False)
        self.clear_files()

    def populate_tree(self):
        """Populate the directory tree with common folders"""
        self.tree_model.clear()
        root_item = self.tree_model.invisibleRootItem()
        
        # Get user's home directory
        home_dir = os.path.expanduser("~")
        
        # Common directories
        common_dirs = {
            'Documents': os.path.join(home_dir, 'Documents'),
            'Pictures': os.path.join(home_dir, 'Pictures'),
            'Downloads': os.path.join(home_dir, 'Downloads'),
            'Desktop': os.path.join(home_dir, 'Desktop')
        }
        
        for name, path in common_dirs.items():
            if os.path.exists(path):
                item = QStandardItem(name)
                item.setData(path, Qt.UserRole)
                root_item.appendRow(item)

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
            self.thread_pool.submit(self.organize_images)
        else:
            self.thread_pool.submit(self.organize_documents)
    def organize_images(self):
        """Organize images using AI classification"""
        if self.file_model.rowCount() == 0:
            QMessageBox.warning(self, "No Files", "Please add some images first.")
            return

        try:
            # Create signals
            signals = WorkerSignals()
            signals.progress.connect(lambda v: self.progress_bar.setValue(v))
            signals.status.connect(lambda msg: self.operation_label.setText(msg))
            signals.finished.connect(self.on_organization_complete)
            signals.error.connect(lambda msg: QMessageBox.critical(self, "Error", msg))

            # Move the GUI updates to the main thread
            self.progress_bar.setVisible(True)
            self.add_button.setEnabled(False)
            self.organize_button.setEnabled(False)

            # Start the worker thread
            self.thread_pool.submit(self._organize_images_worker, signals)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error organizing files: {str(e)}")

    def _organize_images_worker(self, signals):
        """Worker function for image organization"""
        try:
            # Collect image paths
            image_paths = []
            for row in range(self.file_model.rowCount()):
                file_path = self.file_model.item(row).text()
                if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                    image_paths.append(file_path)

            if not image_paths:
                signals.error.emit("No valid image files found.")
                return

            signals.status.emit('Analyzing images with AI...')
            
            # Process images
            results = self.classifier.analyze_images(image_paths)
            num_groups = self.groups_slider.value()
            grouped_images = self.classifier.group_images(results, num_groups=num_groups)

            if grouped_images:
                # Create organized folders and move files
                base_dir = os.path.dirname(image_paths[0])
                organized_dir = os.path.join(base_dir, 'AI_Organized_Images')
                os.makedirs(organized_dir, exist_ok=True)

                for group_id, files in grouped_images.items():
                    group_description = self.classifier.get_group_description(files)
                    group_name = f"Group_{group_id}_{group_description}"
                    group_dir = os.path.join(organized_dir, group_name)
                    os.makedirs(group_dir, exist_ok=True)

                    for file_path in files:
                        new_path = os.path.join(group_dir, os.path.basename(file_path))
                        shutil.copy2(file_path, new_path)

                # Store the results for the completion handler
                self._last_organized_dir = organized_dir
                self._last_organization_stats = (len(image_paths), len(grouped_images))
                
                signals.finished.emit()
            else:
                signals.error.emit("Could not group the images.")

        except Exception as e:
            signals.error.emit(f"Error organizing files: {str(e)}")

    def on_organization_complete(self):
        """Handle completion of organization"""
        organized_dir = getattr(self, '_last_organized_dir', '')
        stats = getattr(self, '_last_organization_stats', (0, 0))
        
        msg = (f"Successfully organized {stats[0]} images into {stats[1]} groups.\n\n"
               f"Location: {organized_dir}\n\n"
               "Would you like to open the folder?")
        
        reply = QMessageBox.question(self, 'Organization Complete', msg,
                                   QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            if os.name == 'nt':
                os.startfile(organized_dir)
            else:
                import subprocess
                subprocess.call(['open', organized_dir])

        self.clear_files()
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
            self.add_button.setEnabled(False)
            self.organize_button.setEnabled(False)
            self.operation_label.setText('')

    def folder_selected(self, index):
        """Handle folder selection"""
        item = self.tree_model.itemFromIndex(index)
        folder_path = item.data(Qt.UserRole)
        if folder_path:
            self.status_bar.showMessage(f'Selected folder: {folder_path}')

    def clear_files(self):
        """Clear the file list"""
        self.file_model.clear()
        self.file_counter_label.setText('Files: 0')
        self.progress_bar.setVisible(False)
        self.operation_label.clear()
        self.status_bar.showMessage('File list cleared')