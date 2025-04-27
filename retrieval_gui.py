#!/usr/bin/env python
import os
import sys
import subprocess
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QComboBox, QPushButton, QLabel, QLineEdit, QFileDialog,
                            QCheckBox, QSpinBox, QDoubleSpinBox, QTabWidget, QScrollArea,
                            QSplitter, QFrame, QGroupBox, QFormLayout, QGraphicsView,
                            QGraphicsScene, QGraphicsPixmapItem, QToolBar, QAction,
                            QDialog, QGridLayout, QTextEdit)
from PyQt5.QtCore import Qt, QSize, QRectF, pyqtSignal
from PyQt5.QtGui import QIcon, QPixmap, QImage, QTransform
from worker_thread import ProcessWorker


class PartsSelectionDialog(QDialog):
    """Dialog for selecting parts with images"""
    
    def __init__(self, parts_info, assembly_id, parent=None):
        super().__init__(parent)
        self.parts_info = parts_info
        self.assembly_id = assembly_id
        self.selected_parts = []
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle(f"Select Parts from Assembly {self.assembly_id}")
        self.setMinimumSize(800, 600)
        
        layout = QVBoxLayout(self)
        
        # Title and instructions
        title = QLabel(f"Assembly {self.assembly_id}: Select Parts for Query")
        title.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(title)
        
        instructions = QLabel("Select the parts you want to include in your query:")
        layout.addWidget(instructions)
        
        # Create scrollable area for parts
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QGridLayout(scroll_content)
        
        # Locate the image directory - search multiple possible locations
        image_dirs = [
            os.path.join("data", "output", "images"),
            os.path.join("data", "images"),
            os.path.join("images"),
            os.path.join("3kdata", "126"),  # Assembly-specific folder
            os.path.join("dataset", "images"),
            # Add additional paths from the workspace structure
            "."  # Current directory as fallback
        ]
        
        # Print debug info to help diagnose image loading issues
        print(f"Looking for images in: {', '.join(image_dirs)}")
        
        image_dir = None
        for dir_path in image_dirs:
            if os.path.exists(dir_path):
                image_dir = dir_path
                print(f"Found image directory: {image_dir}")
                break
        
        # Add parts as selectable items with images
        self.checkboxes = []
        row = 0
        col = 0
        max_cols = 3  # Display 3 parts per row
        
        for i, (filename, part_name) in enumerate(self.parts_info):
            # Create frame for each part
            part_frame = QFrame()
            part_frame.setFrameShape(QFrame.StyledPanel)
            part_layout = QVBoxLayout(part_frame)
            
            # Add image if available
            image_found = False
            if image_dir:
                # Store the assembly ID locally for use within this scope
                current_assembly_id = self.assembly_id
                
                # Try to find the image in the image directory or its parent directories
                possible_paths = [
                    os.path.join(image_dir, filename),
                    os.path.join("data", "output", "images", filename),
                    os.path.join("3kdata", current_assembly_id, filename),
                    # Try without the assembly ID prefix if it's already in the filename
                    os.path.join("3kdata", current_assembly_id, filename.replace(f"{current_assembly_id}_", "")),
                    os.path.join("data", "images", filename)
                ]
                
                # Print debugging info about paths being checked
                print(f"Looking for image {filename} in paths: {possible_paths}")
                
                # Try each possible path
                for img_path in possible_paths:
                    if os.path.exists(img_path):
                        print(f"Found image at: {img_path}")
                        image_label = QLabel()
                        pixmap = QPixmap(img_path)
                        if not pixmap.isNull():
                            pixmap = pixmap.scaled(180, 180, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                            image_label.setPixmap(pixmap)
                            image_label.setAlignment(Qt.AlignCenter)
                            part_layout.addWidget(image_label)
                            image_found = True
                            break
            
            if not image_found:
                # Add placeholder if image not found
                no_image = QLabel("No Image")
                no_image.setAlignment(Qt.AlignCenter)
                no_image.setStyleSheet("background-color: #f0f0f0; min-height: 180px; min-width: 180px;")
                part_layout.addWidget(no_image)
            
            # Add part name and filename
            name_label = QLabel(part_name)
            name_label.setAlignment(Qt.AlignCenter)
            name_label.setStyleSheet("font-weight: bold;")
            part_layout.addWidget(name_label)
            
            filename_label = QLabel(filename)
            filename_label.setAlignment(Qt.AlignCenter)
            filename_label.setWordWrap(True)
            part_layout.addWidget(filename_label)
            
            # Add checkbox for selection
            checkbox = QCheckBox("Select")
            checkbox.setChecked(True)  # Select all by default
            checkbox.setProperty("filename", filename)
            self.checkboxes.append(checkbox)
            part_layout.addWidget(checkbox)
            
            # Add to grid
            scroll_layout.addWidget(part_frame, row, col)
            
            # Update row/column for grid layout
            col += 1
            if col >= max_cols:
                col = 0
                row += 1
        
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
        
        # Button row
        button_row = QHBoxLayout()
        
        select_all = QPushButton("Select All")
        select_all.clicked.connect(self.select_all_parts)
        button_row.addWidget(select_all)
        
        deselect_all = QPushButton("Deselect All")
        deselect_all.clicked.connect(self.deselect_all_parts)
        button_row.addWidget(deselect_all)
        
        button_row.addStretch()
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_row.addWidget(cancel_button)
        
        ok_button = QPushButton("OK")
        ok_button.setDefault(True)
        ok_button.clicked.connect(self.accept)
        button_row.addWidget(ok_button)
        
        layout.addLayout(button_row)
    
    def select_all_parts(self):
        """Select all parts"""
        for checkbox in self.checkboxes:
            checkbox.setChecked(True)
    
    def deselect_all_parts(self):
        """Deselect all parts"""
        for checkbox in self.checkboxes:
            checkbox.setChecked(False)
    
    def get_selected_parts(self):
        """Get list of selected part filenames"""
        selected = []
        for checkbox in self.checkboxes:
            if checkbox.isChecked():
                selected.append(checkbox.property("filename"))
        return selected


class ZoomableGraphicsView(QGraphicsView):
    """Custom graphics view that supports zooming and panning"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHints(self.renderHints() | 1)  # Antialiasing
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.factor = 1.15
        
    def wheelEvent(self, event):
        """Handle mouse wheel events for zooming"""
        if event.angleDelta().y() > 0:
            self.scale(self.factor, self.factor)
        else:
            self.scale(1 / self.factor, 1 / self.factor)

    def reset_view(self):
        """Reset the view to original size"""
        self.resetTransform()
        self.setDragMode(QGraphicsView.ScrollHandDrag)


class ResultTab(QWidget):
    """Tab for displaying a single result image with controls"""
    
    def __init__(self, image_path, similarity=None, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.similarity = similarity
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Create toolbar with controls
        toolbar = QToolBar()
        
        # Add zoom controls
        zoom_in_action = QAction("Zoom In", self)
        zoom_in_action.triggered.connect(self.zoom_in)
        toolbar.addAction(zoom_in_action)
        
        zoom_out_action = QAction("Zoom Out", self)
        zoom_out_action.triggered.connect(self.zoom_out)
        toolbar.addAction(zoom_out_action)
        
        reset_action = QAction("Reset View", self)
        reset_action.triggered.connect(self.reset_view)
        toolbar.addAction(reset_action)
        
        # Add path and similarity info if available
        if self.similarity:
            info_label = QLabel(f"Similarity: {self.similarity:.1f}%")
            toolbar.addWidget(info_label)
            
        path_label = QLabel(f"Path: {self.image_path}")
        path_label.setWordWrap(True)
        
        layout.addWidget(toolbar)
        
        # Create graphics view for the image
        self.view = ZoomableGraphicsView()
        self.scene = QGraphicsScene()
        self.view.setScene(self.scene)
        
        # Load the image
        self.load_image()
        
        layout.addWidget(self.view)
        layout.addWidget(path_label)
        
    def load_image(self):
        if os.path.exists(self.image_path):
            pixmap = QPixmap(self.image_path)
            if not pixmap.isNull():
                pixmap_item = QGraphicsPixmapItem(pixmap)
                self.scene.addItem(pixmap_item)
                self.view.fitInView(pixmap_item, Qt.KeepAspectRatio)
            else:
                self.scene.addText("Failed to load image")
        else:
            self.scene.addText(f"Image not found: {self.image_path}")
    
    def zoom_in(self):
        self.view.scale(1.15, 1.15)
    
    def zoom_out(self):
        self.view.scale(1/1.15, 1/1.15)
        
    def reset_view(self):
        self.view.reset_view()


class ResultThumbnail(QFrame):
    """Clickable thumbnail widget for displaying a result image"""
    
    clicked = pyqtSignal(str, object)
    
    def __init__(self, image_path, info=None, similarity=None, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.info = info
        self.similarity = similarity
        
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        self.setLineWidth(2)
        self.setMinimumWidth(200)
        self.setMinimumHeight(150)
        
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Add image
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        pixmap = QPixmap(self.image_path)
        
        if not pixmap.isNull():
            pixmap = pixmap.scaled(150, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            image_label.setPixmap(pixmap)
        else:
            image_label.setText("No Image")
        
        # Add info if available
        filename = os.path.basename(self.image_path)
        name_label = QLabel(filename)
        name_label.setWordWrap(True)
        name_label.setAlignment(Qt.AlignHCenter)
        name_label.setStyleSheet("font-weight: bold;")
        
        info_text = ""
        if self.info and isinstance(self.info, dict):
            part_name = self.info.get("part_name", "Unknown")
            info_text += f"{part_name}"
        
        if self.similarity is not None:
            info_text += f"\nSimilarity: {self.similarity:.1f}%"
        
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        info_label.setAlignment(Qt.AlignHCenter)
        
        layout.addWidget(image_label)
        layout.addWidget(name_label)
        layout.addWidget(info_label)
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.setFrameShadow(QFrame.Sunken)
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.setFrameShadow(QFrame.Raised)
            self.clicked.emit(self.image_path, self.similarity)


class LeftPanel(QWidget):
    """Left panel containing operation controls"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Operations")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)
        
        # Operation selection
        self.operation_combo = QComboBox()
        self.operation_combo.addItems(["ingest", "train autoencoder", "build", "retrieve"])
        self.operation_combo.currentIndexChanged.connect(self.on_operation_changed)
        
        form_layout = QFormLayout()
        form_layout.addRow("Operation:", self.operation_combo)
        layout.addLayout(form_layout)
        
        # Stack for different operation controls
        self.operation_stack = QWidget()
        layout.addWidget(self.operation_stack)
        self.stack_layout = QVBoxLayout(self.operation_stack)
        
        # Create the initial operation view
        self.create_ingest_view()
        
        # Add run button
        self.run_button = QPushButton("Run")
        layout.addWidget(self.run_button)
        
        # Add log window
        log_group = QGroupBox("Output Log")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(100)
        log_layout.addWidget(self.log_text)
        
        # Add log control buttons
        log_buttons_layout = QHBoxLayout()
        clear_log_button = QPushButton("Clear Log")
        clear_log_button.clicked.connect(self.clear_log)
        save_log_button = QPushButton("Save Log")
        save_log_button.clicked.connect(self.save_log)
        
        log_buttons_layout.addWidget(clear_log_button)
        log_buttons_layout.addWidget(save_log_button)
        log_layout.addLayout(log_buttons_layout)
        
        layout.addWidget(log_group)
        
        # Add some minimal stretch to ensure good layout
        layout.addStretch(1)
    
    def on_operation_changed(self, index):
        """Handle operation selection change"""
        # Clear previous view
        while self.stack_layout.count():
            item = self.stack_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        
        # Create new view based on selection
        if index == 0:  # ingest
            self.create_ingest_view()
        elif index == 1:  # train autoencoder
            self.create_train_view()
        elif index == 2:  # build
            self.create_build_view()
        elif index == 3:  # retrieve
            self.create_retrieve_view()
    
    def create_ingest_view(self):
        """Create controls for the ingest operation"""
        group = QGroupBox("Ingest Data")
        layout = QVBoxLayout(group)
        
        # Dataset directory
        form = QFormLayout()
        self.dataset_dir = QLineEdit()
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_dataset_dir)
        
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(self.dataset_dir)
        dir_layout.addWidget(browse_button)
        form.addRow("Dataset Directory:", dir_layout)
        layout.addLayout(form)
        
        self.stack_layout.addWidget(group)
    
    def create_train_view(self):
        """Create controls for the train autoencoder operation"""
        group = QGroupBox("Train Autoencoder")
        layout = QVBoxLayout(group)
        
        form = QFormLayout()
        
        self.bom_dir = QLineEdit()
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_bom_dir)
        
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(self.bom_dir)
        dir_layout.addWidget(browse_button)
        form.addRow("BOM Directory:", dir_layout)
        
        # Training parameters
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 256)
        self.batch_size.setValue(32)
        form.addRow("Batch Size:", self.batch_size)
        
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 1000)
        self.epochs.setValue(50)
        form.addRow("Epochs:", self.epochs)
        
        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setRange(0.00001, 0.1)
        self.learning_rate.setSingleStep(0.0001)
        self.learning_rate.setValue(0.0001)
        self.learning_rate.setDecimals(6)
        form.addRow("Learning Rate:", self.learning_rate)
        
        self.evaluate_cb = QCheckBox("Evaluate after training")
        form.addRow("", self.evaluate_cb)
        
        self.use_metadata_cb = QCheckBox("Enable metadata integration")
        form.addRow("", self.use_metadata_cb)
        
        layout.addLayout(form)
        self.stack_layout.addWidget(group)
    
    def create_build_view(self):
        """Create controls for the build index operation"""
        group = QGroupBox("Build Index")
        layout = QVBoxLayout(group)
        
        form = QFormLayout()
        
        self.image_dir = QLineEdit()
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_image_dir)
        
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(self.image_dir)
        dir_layout.addWidget(browse_button)
        form.addRow("Image Directory:", dir_layout)
        
        self.build_use_metadata_cb = QCheckBox("Use metadata for indexing")
        form.addRow("", self.build_use_metadata_cb)
        
        layout.addLayout(form)
        self.stack_layout.addWidget(group)
    
    def create_retrieve_view(self):
        """Create controls for the retrieve operation"""
        group = QGroupBox("Retrieve Similar Parts")
        layout = QVBoxLayout(group)
        
        # Query type selection
        form = QFormLayout()
        self.query_type = QComboBox()
        self.query_type.addItems(["Image Query", "Part Name Query", "Full Assembly Query"])
        self.query_type.currentIndexChanged.connect(self.on_query_type_changed)
        form.addRow("Query Type:", self.query_type)
        
        # Stack for query-specific options
        self.query_stack = QWidget()
        query_stack_layout = QVBoxLayout(self.query_stack)
        
        # Image query options
        self.image_query_widget = QWidget()
        image_query_layout = QFormLayout(self.image_query_widget)
        
        self.query_image = QLineEdit()
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_query_image)
        
        query_image_layout = QHBoxLayout()
        query_image_layout.addWidget(self.query_image)
        query_image_layout.addWidget(browse_button)
        image_query_layout.addRow("Query Image:", query_image_layout)
        
        # Part name query options
        self.part_name_widget = QWidget()
        part_name_layout = QFormLayout(self.part_name_widget)
        
        self.part_name = QLineEdit()
        part_name_layout.addRow("Part Name:", self.part_name)
        
        # Full assembly query options
        self.assembly_widget = QWidget()
        assembly_layout = QVBoxLayout(self.assembly_widget)
        
        # Assembly ID input
        id_layout = QFormLayout()
        self.assembly_id = QLineEdit()
        id_layout.addRow("Assembly ID:", self.assembly_id)
        assembly_layout.addLayout(id_layout)
        
        # Add list parts button
        list_parts_button = QPushButton("List Assembly Parts")
        list_parts_button.clicked.connect(self.list_assembly_parts)
        assembly_layout.addWidget(list_parts_button)
        
        # Add parts selection area
        parts_group = QGroupBox("Select Parts for Query")
        parts_layout = QVBoxLayout(parts_group)
        
        # Scrollable area for parts selection
        self.parts_scroll = QScrollArea()
        self.parts_scroll.setWidgetResizable(True)
        self.parts_widget = QWidget()
        self.parts_layout = QVBoxLayout(self.parts_widget)
        self.parts_scroll.setWidget(self.parts_widget)
        self.parts_scroll.setMaximumHeight(150)
        
        parts_layout.addWidget(self.parts_scroll)
        
        # Select/Deselect All buttons
        select_buttons_layout = QHBoxLayout()
        select_all_button = QPushButton("Select All")
        select_all_button.clicked.connect(self.select_all_parts)
        deselect_all_button = QPushButton("Deselect All")
        deselect_all_button.clicked.connect(self.deselect_all_parts)
        
        select_buttons_layout.addWidget(select_all_button)
        select_buttons_layout.addWidget(deselect_all_button)
        parts_layout.addLayout(select_buttons_layout)
        
        assembly_layout.addWidget(parts_group)
        
        # Add the initial widget to the stack
        query_stack_layout.addWidget(self.image_query_widget)
        query_stack_layout.addWidget(self.part_name_widget)
        query_stack_layout.addWidget(self.assembly_widget)
        
        # Initially only show the first query type
        self.part_name_widget.hide()
        self.assembly_widget.hide()
        
        # Common retrieve options
        self.common_options = QWidget()
        common_layout = QFormLayout(self.common_options)
        
        self.k_results = QSpinBox()
        self.k_results.setRange(1, 100)
        self.k_results.setValue(10)
        common_layout.addRow("Number of Results:", self.k_results)
        
        self.visualize_results = QCheckBox("Visualize Results")
        self.visualize_results.setChecked(True)
        common_layout.addRow("", self.visualize_results)
        
        self.rotation_invariant = QCheckBox("Enable Rotation-Invariant Search")
        common_layout.addRow("", self.rotation_invariant)
        
        self.num_rotations = QSpinBox()
        self.num_rotations.setRange(2, 36)
        self.num_rotations.setValue(8)
        self.num_rotations.setEnabled(self.rotation_invariant.isChecked())
        common_layout.addRow("Number of Rotations:", self.num_rotations)
        
        self.rotation_invariant.toggled.connect(lambda checked: self.num_rotations.setEnabled(checked))
        
        self.use_metadata_retrieval = QCheckBox("Use Metadata for Retrieval")
        common_layout.addRow("", self.use_metadata_retrieval)
        
        self.match_threshold = QDoubleSpinBox()
        self.match_threshold.setRange(0.0, 1.0)
        self.match_threshold.setSingleStep(0.05)
        self.match_threshold.setValue(0.7)
        common_layout.addRow("Match Threshold:", self.match_threshold)
        
        # Add the components to the main layout
        layout.addLayout(form)
        layout.addWidget(self.query_stack)
        layout.addWidget(self.common_options)
        
        self.stack_layout.addWidget(group)
    
    def on_query_type_changed(self, index):
        """Handle query type selection change"""
        # Hide all query widgets
        self.image_query_widget.hide()
        self.part_name_widget.hide()
        self.assembly_widget.hide()
        
        # Show the selected query widget
        if index == 0:  # Image Query
            self.image_query_widget.show()
        elif index == 1:  # Part Name Query
            self.part_name_widget.show()
        elif index == 2:  # Full Assembly Query
            self.assembly_widget.show()
    
    def browse_dataset_dir(self):
        """Browse for dataset directory"""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Dataset Directory")
        if dir_path:
            self.dataset_dir.setText(dir_path)
    
    def browse_bom_dir(self):
        """Browse for BOM directory"""
        dir_path = QFileDialog.getExistingDirectory(self, "Select BOM Directory")
        if dir_path:
            self.bom_dir.setText(dir_path)
    
    def browse_image_dir(self):
        """Browse for image directory"""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        if dir_path:
            self.image_dir.setText(dir_path)
    
    def browse_query_image(self):
        """Browse for query image"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Query Image", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            self.query_image.setText(file_path)
    
    def clear_log(self):
        """Clear the log window"""
        self.log_text.clear()
    
    def save_log(self):
        """Save log content to a file"""
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Log", "", "Text Files (*.txt)")
        if file_path:
            with open(file_path, 'w') as f:
                f.write(self.log_text.toPlainText())
    
    def list_assembly_parts(self):
        """List parts for the specified assembly and show them in a popup window"""
        assembly_id = self.assembly_id.text().strip()
        if not assembly_id:
            self.log_text.append("Error: Please enter an Assembly ID")
            return
        
        # Clear previous parts selection
        self.clear_parts_selection()
        
        # Run the command to list assembly parts
        command = ["python", "main.py", "list-assembly-parts", "--assembly-id", assembly_id]
        self.log_text.append(f"Running: {' '.join(command)}")
        
        try:
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode == 0:
                self.log_text.append(result.stdout)
                self.show_parts_selection_dialog(result.stdout, assembly_id)
            else:
                self.log_text.append(f"Error: {result.stderr}")
        except Exception as e:
            self.log_text.append(f"Error: {str(e)}")
    
    def show_parts_selection_dialog(self, output, assembly_id):
        """Show a dialog with images of parts for selection"""
        # Parse the output to extract part filenames and part names
        parts_info = []
        # Print debug info to help diagnose parsing issues
        self.log_text.append(f"Parsing assembly parts for assembly ID: {assembly_id}")
        
        for line in output.split('\n'):
            # Look for lines that list parts (they contain the assembly_id)
            if assembly_id in line and "." in line:
                try:
                    # First, filter out any command outputs that might have been captured accidentally
                    if "python" in line.lower() or "retrieve" in line.lower() or "--select-parts" in line.lower():
                        continue
                        
                    self.log_text.append(f"Parsing line: {line}")
                    
                    # Handle different formats:
                    # Format: "1. 126_SOLID.png - SOLID"
                    if " - " in line:
                        # Split by the first period to separate index from content
                        parts = line.split(".", 1)
                        if len(parts) > 1:
                            content = parts[1].strip()
                            # Then split by the dash to get filename and part name
                            if " - " in content:
                                filename_part, part_name = content.split(" - ", 1)
                                filename = filename_part.strip()
                                part_name = part_name.strip()
                            else:
                                filename = content.strip()
                                part_name = "Unknown"
                    
                    # Format: "1. 126_pin.png" (without part name)
                    else:
                        parts = line.split(".", 1)
                        if len(parts) > 1:
                            filename = parts[1].strip()
                            # Try to extract part name from filename
                            if "_" in filename:
                                # Extract part name from filename (e.g. "126_SOLID.png" -> "SOLID")
                                base_name = os.path.splitext(filename)[0]  # Remove extension
                                part_name = base_name.split("_", 1)[1] if "_" in base_name else "Unknown"
                            else:
                                part_name = "Unknown"
                    
                    # Only add if we successfully extracted a filename with the assembly ID
                    if filename and assembly_id in filename and any(ext in filename.lower() for ext in ['.png', '.jpg', '.jpeg']):
                        parts_info.append((filename, part_name))
                        self.log_text.append(f"Added part: {filename} - {part_name}")
                    else:
                        self.log_text.append(f"Skipped line due to missing assembly ID or valid extension: {filename}")
                        
                except Exception as e:
                    self.log_text.append(f"Error parsing part info from line '{line}': {str(e)}")
        
        if not parts_info:
            self.log_text.append("No parts found or could not parse part names from output.")
            return
        
        # Create a dialog to display parts with images
        dialog = PartsSelectionDialog(parts_info, assembly_id, self)
        if dialog.exec_():
            # User confirmed selection
            selected_parts = dialog.get_selected_parts()
            if selected_parts:
                self.log_text.append(f"Selected {len(selected_parts)} parts for assembly {assembly_id}")
                # Store selected parts for later use
                self.selected_parts = selected_parts
                # Display the selected parts in the main window
                self.update_parts_display(selected_parts)
            else:
                self.log_text.append("No parts were selected.")
        else:
            self.log_text.append("Parts selection canceled.")
    
    def update_parts_display(self, selected_parts):
        """Update the parts display in the main window with selected parts"""
        self.clear_parts_selection()
        
        for part in selected_parts:
            checkbox = QCheckBox(part)
            checkbox.setChecked(True)
            self.parts_layout.addWidget(checkbox)
    
    def get_selected_parts(self):
        """Get the list of selected parts"""
        selected_parts = []
        for i in range(self.parts_layout.count()):
            widget = self.parts_layout.itemAt(i).widget()
            if isinstance(widget, QCheckBox) and widget.isChecked():
                selected_parts.append(widget.text())
        return selected_parts
    
    def clear_parts_selection(self):
        """Clear the parts selection area"""
        while self.parts_layout.count():
            item = self.parts_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
    
    def select_all_parts(self):
        """Select all parts in the list"""
        for i in range(self.parts_layout.count()):
            widget = self.parts_layout.itemAt(i).widget()
            if isinstance(widget, QCheckBox):
                widget.setChecked(True)
    
    def deselect_all_parts(self):
        """Deselect all parts in the list"""
        for i in range(self.parts_layout.count()):
            widget = self.parts_layout.itemAt(i).widget()
            if isinstance(widget, QCheckBox):
                widget.setChecked(False)


class RightPanel(QWidget):
    """Right panel containing result thumbnails"""
    
    thumbnail_clicked = pyqtSignal(str, object)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Results")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)
        
        # Result type selection
        self.result_combo = QComboBox()
        self.result_combo.addItems(["All Results", "Image Query Results", "Part Name Results", "Full Assembly Results"])
        self.result_combo.currentIndexChanged.connect(self.filter_results)
        
        form_layout = QFormLayout()
        form_layout.addRow("Display:", self.result_combo)
        layout.addLayout(form_layout)
        
        # Scrollable area for thumbnails
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        self.thumbnails_widget = QWidget()
        self.thumbnails_layout = QVBoxLayout(self.thumbnails_widget)
        self.thumbnails_layout.setAlignment(Qt.AlignTop)
        
        self.scroll_area.setWidget(self.thumbnails_widget)
        layout.addWidget(self.scroll_area)
        
        # Refresh button
        self.refresh_button = QPushButton("Refresh Results")
        self.refresh_button.clicked.connect(self.refresh_results)
        layout.addWidget(self.refresh_button)
    
    def add_thumbnail(self, image_path, info=None, similarity=None, result_type="image"):
        """Add a thumbnail to the panel"""
        thumbnail = ResultThumbnail(image_path, info, similarity)
        thumbnail.clicked.connect(self.thumbnail_clicked)
        thumbnail.setProperty("result_type", result_type)  # Store result type for filtering
        self.thumbnails_layout.addWidget(thumbnail)
    
    def clear_thumbnails(self):
        """Clear all thumbnails"""
        while self.thumbnails_layout.count():
            item = self.thumbnails_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
    
    def filter_results(self, index):
        """Filter results by type"""
        result_type = self.result_combo.currentText()
        
        # Show/hide thumbnails based on selected type
        for i in range(self.thumbnails_layout.count()):
            item = self.thumbnails_layout.itemAt(i)
            widget = item.widget()
            if widget:
                if result_type == "All Results" or widget.property("result_type") in result_type:
                    widget.show()
                else:
                    widget.hide()
    
    def refresh_results(self):
        """Refresh the results from the filesystem"""
        self.clear_thumbnails()        # Check if the results directories exist
        result_dirs = {
            "Image Query Results": "image_queries",
            "Part Name Results": "name_searches",
            "Full Assembly Results": "full_assembly_queries"
        }
        
        for result_type, dir_name in result_dirs.items():
            # Try in data/output/results directory (correct path)
            dir_path = os.path.join("data", "output", "results", dir_name)
            # If not found, try at root or in data directory as fallback
            if not os.path.exists(dir_path):
                if os.path.exists(dir_name):
                    dir_path = dir_name
                elif os.path.exists(os.path.join("data", dir_name)):
                    dir_path = os.path.join("data", dir_name)
                else:
                    continue
                
            # Get all image files in the directory
            for file in os.listdir(dir_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(dir_path, file)
                    self.add_thumbnail(image_path, result_type=result_type)
        
        # Filter based on current selection
        self.filter_results(self.result_combo.currentIndex())


class RetrievalGUI(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Part Retrieval System")
        self.setMinimumSize(1200, 800)
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # Create splitter for resizable panels
        self.splitter = QSplitter(Qt.Horizontal)
        
        # Left panel
        self.left_panel = LeftPanel()
        self.splitter.addWidget(self.left_panel)
        
        # Center panel with tabs
        self.center_panel = QTabWidget()
        self.center_panel.setTabsClosable(True)
        self.center_panel.tabCloseRequested.connect(self.close_tab)
        self.splitter.addWidget(self.center_panel)
        
        # Welcome tab
        welcome_widget = QWidget()
        welcome_layout = QVBoxLayout(welcome_widget)
        welcome_label = QLabel("Welcome to the Part Retrieval System")
        welcome_label.setAlignment(Qt.AlignCenter)
        welcome_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        
        instructions = QLabel(
            "1. Use the left panel to select an operation\n"
            "2. Configure the operation parameters\n"
            "3. Click 'Run' to execute the operation\n"
            "4. Results will appear in the right panel\n"
            "5. Click on a result thumbnail to view it in a tab"
        )
        instructions.setAlignment(Qt.AlignCenter)
        
        welcome_layout.addStretch()
        welcome_layout.addWidget(welcome_label)
        welcome_layout.addWidget(instructions)
        welcome_layout.addStretch()
        
        self.center_panel.addTab(welcome_widget, "Welcome")
        
        # Right panel
        self.right_panel = RightPanel()
        self.right_panel.thumbnail_clicked.connect(self.open_result_tab)
        self.splitter.addWidget(self.right_panel)
        
        # Set the splitter sizes
        self.splitter.setSizes([300, 600, 300])
        
        main_layout.addWidget(self.splitter)
        
        # Connect the run button
        self.left_panel.run_button.clicked.connect(self.run_operation)
        
        # Load initial results
        self.right_panel.refresh_results()
        
        self.statusBar().showMessage("Ready")
    
    def open_result_tab(self, image_path, similarity=None):
        """Open an image in a new tab"""
        tab_name = os.path.basename(image_path)
        
        # Create a result tab for the image
        result_tab = ResultTab(image_path, similarity)
        
        # Add tab
        index = self.center_panel.addTab(result_tab, tab_name)
        self.center_panel.setCurrentIndex(index)
    
    def close_tab(self, index):
        """Close a tab"""
        # Don't close the welcome tab (index 0)
        if index > 0:
            self.center_panel.removeTab(index)
    
    def run_operation(self):
        """Run the selected operation"""
        operation = self.left_panel.operation_combo.currentText()
        command = ["python", "main.py"]
        
        # Add operation-specific arguments
        if operation == "ingest":
            dataset_dir = self.left_panel.dataset_dir.text()
            if not dataset_dir:
                self.statusBar().showMessage("Error: Dataset directory is required", 5000)
                return
                
            command.extend(["ingest", "--dataset_dir", dataset_dir])
            
        elif operation == "train autoencoder":
            bom_dir = self.left_panel.bom_dir.text()
            if not bom_dir:
                self.statusBar().showMessage("Error: BOM directory is required", 5000)
                return
                
            command.extend(["train-autoencoder", "--bom_dir", bom_dir])
            
            # Add training parameters
            command.extend(["--batch_size", str(self.left_panel.batch_size.value())])
            command.extend(["--epochs", str(self.left_panel.epochs.value())])
            command.extend(["--lr", str(self.left_panel.learning_rate.value())])
            
            if self.left_panel.evaluate_cb.isChecked():
                command.append("--evaluate")
                
            if self.left_panel.use_metadata_cb.isChecked():
                command.append("--use-metadata")
                
        elif operation == "build":
            image_dir = self.left_panel.image_dir.text()
            if not image_dir:
                self.statusBar().showMessage("Error: Image directory is required", 5000)
                return
                
            command.extend(["build", "--image_dir", image_dir])
            
            if self.left_panel.build_use_metadata_cb.isChecked():
                command.append("--use-metadata")
                
        elif operation == "retrieve":
            query_type = self.left_panel.query_type.currentText()
            
            if query_type == "Image Query":
                query_image = self.left_panel.query_image.text()
                if not query_image:
                    self.statusBar().showMessage("Error: Query image is required", 5000)
                    return
                    
                command.extend(["retrieve", "--query", query_image])
                
            elif query_type == "Part Name Query":
                part_name = self.left_panel.part_name.text()
                if not part_name:
                    self.statusBar().showMessage("Error: Part name is required", 5000)
                    return
                    
                command.extend(["retrieve", "--part-name", part_name])
                
            elif query_type == "Full Assembly Query":
                assembly_id = self.left_panel.assembly_id.text()
                if not assembly_id:
                    self.statusBar().showMessage("Error: Assembly ID is required", 5000)
                    self.left_panel.log_text.append("Error: Assembly ID is required")
                    return
                    
                command.extend(["retrieve", "--full-assembly", assembly_id])
                
                # Check if there are selected parts to include
                selected_parts = self.left_panel.get_selected_parts()
                if selected_parts:
                    command.append("--select-parts")
                    command.extend(selected_parts)
                    self.left_panel.log_text.append(f"Using {len(selected_parts)} selected parts for assembly search")
            
            # Common retrieve options
            command.extend(["--k", str(self.left_panel.k_results.value())])
            
            if self.left_panel.visualize_results.isChecked():
                command.append("--visualize")
                
            if self.left_panel.rotation_invariant.isChecked():
                command.append("--rotation-invariant")
                command.extend(["--num-rotations", str(self.left_panel.num_rotations.value())])
                
            if self.left_panel.use_metadata_retrieval.isChecked():
                command.append("--use-metadata")
                
            command.extend(["--match-threshold", str(self.left_panel.match_threshold.value())])
        
        # Display the command in the log
        self.statusBar().showMessage(f"Running: {' '.join(command)}")
        self.left_panel.log_text.append(f"Running: {' '.join(command)}")
        
        # Disable the run button to prevent multiple clicks
        self.left_panel.run_button.setEnabled(False)
        
        # Create a worker thread to run the command
        self.worker = ProcessWorker(command)
        
        # Connect signals
        self.worker.output_ready.connect(lambda line: self.left_panel.log_text.append(line))
        # Handle stderr output without adding "Error:" for progress bars and other non-error output
        self.worker.error_ready.connect(lambda line: self.left_panel.log_text.append(
            # Don't add "Error:" prefix for progress bars, empty lines, or known informational outputs
            line if (
                # Progress bar detection
                ('%' in line and ('|' in line or '[' in line)) or
                # Empty lines
                line.strip() == '' or
                # Lines already containing error-related terms
                any(term in line.lower() for term in ["error", "warning", "exception", "traceback"])
            ) else f"Error: {line}")
        )
        self.worker.process_finished.connect(self.on_process_finished)
        
        # Start the worker thread
        self.worker.start()
    
    def on_process_finished(self, return_code):
        """Handle process completion"""
        # Re-enable the run button
        self.left_panel.run_button.setEnabled(True)
        
        if return_code == 0:
            self.statusBar().showMessage("Operation completed successfully")
            self.left_panel.log_text.append("Operation completed successfully")
            # Refresh results
            self.right_panel.refresh_results()
        else:
            self.statusBar().showMessage(f"Operation failed with return code {return_code}")
            self.left_panel.log_text.append(f"Operation failed with return code {return_code}")


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Use Fusion style for a modern look
    
    # Set stylesheet for a more attractive appearance
    app.setStyleSheet("""
    QMainWindow {
        background-color: #f0f0f0;
    }
    
    QSplitter::handle {
        background-color: #dddddd;
    }
    
    QTabWidget::pane {
        border: 1px solid #cccccc;
        background-color: white;
    }
    
    QTabBar::tab {
        background-color: #e0e0e0;
        border: 1px solid #c0c0c0;
        padding: 6px 12px;
        margin-right: 2px;
    }
    
    QTabBar::tab:selected {
        background-color: white;
        border-bottom-color: white;
    }
    
    QPushButton {
        background-color: #1565c0;
        color: white;
        padding: 5px 15px;
        border: none;
        border-radius: 3px;
    }
    
    QPushButton:hover {
        background-color: #1976d2;
    }
    
    QPushButton:pressed {
        background-color: #0d47a1;
    }
    
    QGroupBox {
        border: 1px solid #cccccc;
        border-radius: 3px;
        margin-top: 1ex;
    }
    
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top center;
        padding: 0 5px;
    }
    
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
        padding: 4px;
        border: 1px solid #cccccc;
        border-radius: 2px;
    }
    """)
    
    window = RetrievalGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
