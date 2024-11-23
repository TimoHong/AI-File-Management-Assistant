import os
import sys
import shutil
import logging
from concurrent.futures import ThreadPoolExecutor
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QMessageBox

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
from PyQt5.QtCore import Qt, QSize, QTimer, pyqtSignal, QObject, QMetaObject, Q_ARG, QThread, pyqtSlot
from PyQt5.QtGui import QIcon, QStandardItemModel, QStandardItem

class WorkerSignals(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    status = pyqtSignal(str)

class MainWindow(QMainWindow):
    # Add these signals at the class level
    update_progress_signal = pyqtSignal(int)
    update_status_signal = pyqtSignal(str)
    show_error_signal = pyqtSignal(str)
    show_completion_dialog = pyqtSignal(tuple)
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.files_to_process = []
        self.current_progress = 0
        self.classifier = None
        self.doc_classifier = None
        self.thread_pool = ThreadPoolExecutor(max_workers=1)
        self.setup_classifiers()
        
        # Connect signals to slots using Qt.QueuedConnection
        self.update_progress_signal.connect(self.update_progress_bar, Qt.QueuedConnection)
        self.update_status_signal.connect(self._update_status, Qt.QueuedConnection)
        self.show_error_signal.connect(self._show_error, Qt.QueuedConnection)
        self.show_completion_dialog.connect(self._show_completion_dialog, Qt.QueuedConnection)

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
        self.groups_slider.valueChanged.connect(self.update_groups_value)

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
        else:
            self.status_bar.showMessage('Switched to Document Mode')
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
            # Collect image paths
            image_paths = []
            for row in range(self.file_model.rowCount()):
                file_path = self.file_model.item(row).text()
                if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                    image_paths.append(file_path)

            if not image_paths:
                QMessageBox.warning(self, "No Images", "No valid image files found.")
                return

            # Store paths as instance variable
            self.image_paths = image_paths

            # Update GUI state
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.add_button.setEnabled(False)
            self.organize_button.setEnabled(False)

            # Create and setup worker thread
            self.image_worker = ImageWorkerThread(
                self.classifier, 
                image_paths, 
                self.groups_slider.value()
            )
            
            # Connect signals using Qt.QueuedConnection
            self.image_worker.progress.connect(self.update_progress_bar, Qt.QueuedConnection)
            self.image_worker.status.connect(self._update_status, Qt.QueuedConnection)
            self.image_worker.error.connect(self._show_error, Qt.QueuedConnection)
            self.image_worker.results_ready.connect(self.process_image_results, Qt.QueuedConnection)
            self.image_worker.finished.connect(self.on_organization_complete, Qt.QueuedConnection)
            
            # Start the thread
            self.image_worker.start()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error organizing files: {str(e)}")

    def process_image_results(self, data):
        """Process image results in the main thread"""
        try:
            results, grouped_images = data
            if grouped_images:
                base_dir = os.path.dirname(self.image_paths[0])
                organized_dir = os.path.join(base_dir, 'AI_Organized_Images')
                os.makedirs(organized_dir, exist_ok=True)

                import shutil
                total_files = 0

                for group_name, files in grouped_images.items():
                    group_dir = os.path.join(organized_dir, group_name)
                    os.makedirs(group_dir, exist_ok=True)
                    total_files += len(files)

                    for file_path in files:
                        try:
                            if os.path.exists(file_path):
                                dest_path = os.path.join(group_dir, os.path.basename(file_path))
                                shutil.copy2(file_path, dest_path)
                        except Exception as e:
                            logging.error(f"Error copying file {file_path}: {str(e)}")

                # Show completion dialog
                msg = (f"Successfully organized {total_files} images into {len(grouped_images)} categories.\n\n"
                      f"Location: {organized_dir}\n\n"
                      "Would you like to open the folder?")
                
                reply = QMessageBox.question(self, 'Organization Complete', msg,
                                           QMessageBox.Yes | QMessageBox.No)
                
                if reply == QMessageBox.Yes:
                    if os.name == 'nt':  # Windows
                        os.startfile(organized_dir)
                    else:  # macOS and Linux
                        import subprocess
                        subprocess.run(['xdg-open' if os.name == 'posix' else 'open', organized_dir])

                # Reset GUI state
                self.clear_files()
                self.progress_bar.setVisible(False)
                self.add_button.setEnabled(True)
                self.organize_button.setEnabled(True)
                self.operation_label.setText('')

            else:
                self.show_error_signal.emit("Could not group the images.")

        except Exception as e:
            self.show_error_signal.emit(f"Error organizing images: {str(e)}")

    @pyqtSlot()
    def on_organization_complete(self):
        """Handle completion of organization"""
        try:
            # Reset GUI state
            self.progress_bar.setVisible(False)
            self.add_button.setEnabled(True)
            self.organize_button.setEnabled(True)
            self.operation_label.setText('')
            
        except Exception as e:
            logging.error(f"Error in completion handler: {str(e)}")
            self.show_error_signal.emit(f"Error in completion handler: {str(e)}")

    @pyqtSlot(int)
    def update_progress_bar(self, value):
        """Update progress bar from any thread"""
        if self.progress_bar:
            self.progress_bar.setValue(value)

    @pyqtSlot(str)
    def _update_status(self, message):
        """Update status label from any thread"""
        if self.operation_label:
            self.operation_label.setText(message)

    @pyqtSlot(str)
    def _show_error(self, message):
        """Show error message from any thread"""
        QMessageBox.critical(self, "Error", message)

    def organize_documents(self):
        """Organize documents using AI classification"""
        if self.file_model.rowCount() == 0:
            QMessageBox.warning(self, "No Files", "Please add some documents first.")
            return

        try:
            # Collect document paths
            doc_paths = []
            for row in range(self.file_model.rowCount()):
                file_path = self.file_model.item(row).text()
                if file_path.lower().endswith('.pdf'):
                    doc_paths.append(file_path)

            if not doc_paths:
                QMessageBox.warning(self, "No Documents", "No valid PDF documents found.")
                return

            # Store paths as instance variable
            self.doc_paths = doc_paths

            # Update GUI state
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.add_button.setEnabled(False)
            self.organize_button.setEnabled(False)

            # Create and setup worker thread
            self.doc_worker = DocumentWorkerThread(
                self.doc_classifier, 
                doc_paths, 
                self.groups_slider.value()
            )
            
            # Connect signals using Qt.QueuedConnection
            self.doc_worker.progress.connect(self.update_progress_bar, Qt.QueuedConnection)
            self.doc_worker.status.connect(self._update_status, Qt.QueuedConnection)
            self.doc_worker.error.connect(self._show_error, Qt.QueuedConnection)
            self.doc_worker.results_ready.connect(self.process_document_results, Qt.QueuedConnection)
            self.doc_worker.finished.connect(self.on_organization_complete, Qt.QueuedConnection)
            
            # Start the thread
            self.doc_worker.start()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error organizing documents: {str(e)}")

    @pyqtSlot(tuple)
    def process_document_results(self, data):
        """Process document results in the main thread"""
        try:
            results, grouped_docs = data
            if grouped_docs:
                base_dir = os.path.dirname(self.doc_paths[0])
                organized_dir = os.path.join(base_dir, 'AI_Organized_Documents')
                os.makedirs(organized_dir, exist_ok=True)

                total_files = 0
                for group_name, files in grouped_docs.items():
                    safe_group_name = group_name.replace('/', '_').replace('\\', '_')
                    group_dir = os.path.join(organized_dir, safe_group_name)
                    os.makedirs(group_dir, exist_ok=True)
                    total_files += len(files)

                    for file_path in files:
                        try:
                            if os.path.exists(file_path):
                                dest_path = os.path.join(group_dir, os.path.basename(file_path))
                                shutil.copy2(file_path, dest_path)
                        except Exception as e:
                            logging.error(f"Error copying file {file_path}: {str(e)}")

                # Show completion dialog
                msg = (f"Successfully organized {total_files} documents into {len(grouped_docs)} categories.\n\n"
                      f"Location: {organized_dir}\n\n"
                      "Would you like to open the folder?")
                
                # Use signal to show dialog in main thread
                self.show_completion_dialog.emit((organized_dir, msg))

            else:
                self.show_error_signal.emit("Could not group the documents.")

        except Exception as e:
            logging.error(f"Error in process_document_results: {str(e)}")
            self.show_error_signal.emit(f"Error organizing documents: {str(e)}")

    @pyqtSlot(tuple)
    def _show_completion_dialog(self, data):
        """Show completion dialog in the main thread"""
        organized_dir, msg = data
        reply = QMessageBox.question(self, 'Organization Complete', msg,
                                   QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            if os.name == 'nt':  # Windows
                os.startfile(organized_dir)
            else:  # macOS and Linux
                import subprocess
                subprocess.run(['xdg-open' if os.name == 'posix' else 'open', organized_dir])

        # Reset GUI state
        self.clear_files()
        self.progress_bar.setVisible(False)
        self.add_button.setEnabled(True)
        self.organize_button.setEnabled(True)
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

    def closeEvent(self, event):
        """Clean up when closing the window"""
        if hasattr(self, 'image_worker'):
            self.image_worker.stop()
            self.image_worker.wait()
        if hasattr(self, 'doc_worker'):
            self.doc_worker.stop()
            self.doc_worker.wait()
        event.accept()

    def process_document(self, text):
        # Direct call without threading
        try:
            results = self.doc_classifier.process_document(text)
            self.update_results(results)  # Direct update of UI
        except Exception as e:
            print(f"Error processing document: {str(e)}")
    
    def update_results(self, results):
        # Update your UI directly here
        self.status_bar.showMessage(f"Results updated: {len(results)} documents processed")
        self.file_model.clear()
        for result in results:
            item = QStandardItem(result['filename'])
            item.setData(result['path'], Qt.UserRole)
            self.file_model.appendRow(item)
        self.file_counter_label.setText(f'Files: {len(results)}')

class DocumentWorkerThread(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    error = pyqtSignal(str)
    results_ready = pyqtSignal(tuple)

    def __init__(self, classifier, doc_paths, num_groups):
        super().__init__()
        self.classifier = classifier
        self.doc_paths = doc_paths
        self.num_groups = num_groups
        self._is_running = True

    def run(self):
        try:
            self.status.emit("Analyzing documents...")
            total_docs = len(self.doc_paths)
            
            if total_docs == 0:
                self.error.emit("No documents to process.")
                return
            
            # Analyze all documents first
            results = self.classifier.analyze_documents(self.doc_paths)
            if not results:
                self.error.emit("Could not analyze documents.")
                return

            self.progress.emit(50)
            
            # Use the document classifier's grouping function with themes
            grouped_docs = self.classifier.group_documents(
                results, 
                confidence_threshold=0.5,
                num_groups=self.num_groups
            )
            
            if not grouped_docs:
                self.error.emit("Could not group the documents.")
                return
                
            self.progress.emit(90)
            self.results_ready.emit((results, grouped_docs))
            self.progress.emit(100)
            
        except Exception as e:
            logging.error(f"Error in document classification: {str(e)}")
            self.error.emit(f"Error in document classification: {str(e)}")
        finally:
            if self._is_running:
                self.finished.emit()

    def stop(self):
        self._is_running = False

class ImageWorkerThread(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    results_ready = pyqtSignal(tuple)  # Changed from object to tuple

    def __init__(self, classifier, image_paths, num_groups):
        super().__init__()
        self.classifier = classifier
        self.image_paths = image_paths
        self.num_groups = num_groups
        self._is_running = True

    def run(self):
        try:
            self.status.emit('Analyzing images with AI...')
            self.progress.emit(10)
            
            results = self.classifier.analyze_images(self.image_paths)
            if not results or not self._is_running:
                self.error.emit("Could not analyze images.")
                return
                
            self.progress.emit(50)
            self.status.emit('Grouping images...')
            
            grouped_images = self.classifier.group_images(results, num_groups=self.num_groups)
            if not grouped_images or not self._is_running:
                self.error.emit("Could not group the images.")
                return
                
            self.progress.emit(90)
            self.results_ready.emit((results, grouped_images))
            self.progress.emit(100)
            
        except Exception as e:
            logging.error(f"Error in image worker thread: {str(e)}")
            self.error.emit(f"Error organizing files: {str(e)}")
        finally:
            if self._is_running:
                self.finished.emit()

    def stop(self):
        self._is_running = False