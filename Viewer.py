import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QPushButton, QVBoxLayout, 
    QHBoxLayout, QWidget, QTreeWidget, QTreeWidgetItem, QGroupBox, 
    QLabel, QSlider, QComboBox, QMessageBox, QMenuBar, QMenu, QAction, 
    QLineEdit, QToolBar, QRadioButton, QColorDialog, QInputDialog,
    QDockWidget, QVBoxLayout, QLabel, QLineEdit, QComboBox, QRadioButton, 
    QButtonGroup, QPushButton, QHBoxLayout, QWidget, QMessageBox, QComboBox,QListWidget
)
from PyQt5.QtCore import Qt
import vtk

# class CustomInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
#     def __init__(self, parent=None):
#         super().__init__()
#         self.AddObserver("RightButtonPressEvent", self.on_right_button_press)

#     def on_right_button_press(self, obj, event):
#         pass

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voxel Viewer")
        self.setGeometry(100, 100, 1000, 800)

        # Menu Bar
        self.create_menu_bar()

        # Tool Bar
        self.create_tool_bar()

        main_layout = QHBoxLayout()

        left_layout = QVBoxLayout()
        self.create_model_tree(left_layout)
        self.create_model_controls(left_layout)
        main_layout.addLayout(left_layout, 2)

        self.plotter_widget = QtInteractor(self)
        main_layout.addWidget(self.plotter_widget, 5)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.models = {} 

        self.tree_widget.itemChanged.connect(self.on_tree_item_changed)

        self.plotter_widget.add_axes()
    #     self.setup_interactor()

    # def setup_interactor(self):
    #     interactor = self.plotter_widget.interactor
    #     custom_style = CustomInteractorStyle()
    #     interactor.SetInteractorStyle(custom_style)

    def create_menu_bar(self):
        menubar = self.menuBar()

        # File Menu
        file_menu = menubar.addMenu('File')
        open_action = QAction('Open', self)
        open_action.triggered.connect(self.load_file)
        file_menu.addAction(open_action)

        save_action = QAction('Save', self)
        save_action.triggered.connect(self.save_file)
        file_menu.addAction(save_action)

        # Tool Menu
        tool_menu = menubar.addMenu('Tool')
        add_cell_action = QAction('Add CELL', self)
        add_cell_action.triggered.connect(self.add_cell)
        tool_menu.addAction(add_cell_action)

        add_point_action = QAction('Add POINT', self)
        add_point_action.triggered.connect(self.add_point)
        tool_menu.addAction(add_point_action)

        # Work Menu
        work_menu = menubar.addMenu('Work')
        voxelize_action = QAction('Voxelizer', self)
        voxelize_action.triggered.connect(self.voxelize_model)
        work_menu.addAction(voxelize_action)

    def create_tool_bar(self):
        toolbar = QToolBar("Toolbar")
        self.addToolBar(toolbar)

        # View Buttons
        views = [
            ("Adaptive View", self.reset_camera),
            ("Front View", lambda: self.set_camera_position("front")),
            ("Back View", lambda: self.set_camera_position("back")),
            ("Left View", lambda: self.set_camera_position("left")),
            ("Right View", lambda: self.set_camera_position("right")),
            ("Top View", lambda: self.set_camera_position("top")),
            ("Bottom View", lambda: self.set_camera_position("bottom")),
        ]
        for name, func in views:
            btn = QPushButton(name)
            btn.clicked.connect(func)
            toolbar.addWidget(btn)

        # Edge Display Mode
        display_mode_label = QLabel("Display Mode:")
        self.display_mode_combo = QComboBox()
        self.display_mode_combo.addItems(["with Edge", "without Edge"])
        self.display_mode_combo.currentIndexChanged.connect(self.update_display_mode)
        toolbar.addWidget(display_mode_label)
        toolbar.addWidget(self.display_mode_combo)

    def update_display_mode(self):
        display_mode = self.display_mode_combo.currentText()
        for file_name, (mesh, _, actor) in self.models.items():
            if display_mode == "with Edge":
                actor.GetProperty().SetRepresentationToSurface()
                actor.GetProperty().EdgeVisibilityOn()
            elif display_mode == "without Edge":
                actor.GetProperty().SetRepresentationToSurface()
                actor.GetProperty().EdgeVisibilityOff()

        self.plotter_widget.update()

    def create_model_tree(self, layout):
        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabel("Loaded Models")
        layout.addWidget(self.tree_widget, 5)

    def create_model_controls(self, layout):
        control_group = QGroupBox("Model Controls")
        control_layout = QVBoxLayout()

        opacity_label = QLabel("Opacity:")
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.valueChanged.connect(self.update_opacity)
        control_layout.addWidget(opacity_label)
        control_layout.addWidget(self.opacity_slider)

        colormap_label = QLabel("Colormap:")
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(["viridis", "plasma", "coolwarm", "gray"])
        self.colormap_combo.currentIndexChanged.connect(self.update_colormap)
        control_layout.addWidget(colormap_label)
        control_layout.addWidget(self.colormap_combo)

        threshold_label = QLabel("Threshold:")
        threshold_layout = QHBoxLayout()
        self.min_threshold = QLineEdit("0.0")
        self.max_threshold = QLineEdit("1.0")
        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self.update_threshold)
        threshold_layout.addWidget(self.min_threshold)
        threshold_layout.addWidget(QLabel("~"))
        threshold_layout.addWidget(self.max_threshold)
        threshold_layout.addWidget(apply_button)
        control_layout.addLayout(threshold_layout)

        control_group.setLayout(control_layout)
        layout.addWidget(control_group, 3)

    def load_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Open File", 
            "", 
            "VTK Files (*.vtk);;STL Files (*.stl)", 
            options=options
        )
        if file_path:
            self.add_model(file_path)

    def add_model(self, file_path):
        try:
            print(f"Reading file: {file_path}")
            mesh = pv.read(file_path)
            print("File read successfully!")

            file_name = file_path.split("/")[-1]
            if file_name in self.models:
                QMessageBox.warning(self, "Duplicate File", f"Model '{file_name}' is already loaded.")
                return

            tree_item = QTreeWidgetItem(self.tree_widget)
            tree_item.setText(0, file_name)
            tree_item.setCheckState(0, 2)
            self.tree_widget.addTopLevelItem(tree_item)

            actor = self.plotter_widget.add_mesh(mesh, name=file_name, show_edges=True)
            self.plotter_widget.reset_camera()

            self.models[file_name] = (mesh, tree_item, actor)
            self.update_display_mode()
        except FileNotFoundError:
            QMessageBox.critical(self, "Error", f"File not found: {file_path}")
        except ValueError as e:
            QMessageBox.critical(self, "Error", f"Invalid file format: {e}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file: {e}")

    def save_file(self):
        dock = QDockWidget(f"Save Model", self)
        dock.setFloating(True)
        dock.setGeometry(200, 200, 300, 300)
        menu_widget = QWidget()
        menu_layout = QVBoxLayout()
        menu_widget.setLayout(menu_layout)
        dock.setWidget(menu_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        dock.setFocusPolicy(Qt.StrongFocus)
        dock.activateWindow()
        dock.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)

        model_label = QLabel("Select Model:")
        model_combo = QComboBox()
        model_combo.addItems(self.models.keys())
        menu_layout.addWidget(model_label)
        menu_layout.addWidget(model_combo)
        
        file_name_label = QLabel("File Name:")
        file_name_input = QLineEdit()
        menu_layout.addWidget(file_name_label)
        menu_layout.addWidget(file_name_input)

        save_button = QPushButton("Save")
        save_button.clicked.connect(lambda: self.save_model(model_combo.currentText(), file_name_input.text()))
        menu_layout.addWidget(save_button)

    def save_model(self, model_name, file_name):
        if model_name in self.models:
            mesh, _, _ = self.models[model_name]
            if file_name:
                try:
                    mesh.save(file_name)
                    
                    QMessageBox.information(self, "File Saved", f"Model '{model_name}' saved to '{file_name}'.")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to save file: {e}")
            else:
                QMessageBox.warning(self, "Invalid File Name", "Please enter a valid file name.")
        else:
            QMessageBox.warning(self, "Invalid Model", "Please select a valid model to save.")

    def add_cell(self):

        pass

    def add_point(self):
        dock = QDockWidget("Add Sets", self)
        dock.setFloating(True)
        dock.setGeometry(200, 200, 300, 300)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        dock.setFocusPolicy(Qt.StrongFocus)
        dock.activateWindow()
        dock.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)

        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        label = QLabel("Click to pick points:")
        layout.addWidget(label)

        finish_button = QPushButton("Finish Selection")
        layout.addWidget(finish_button)

        self.picked_points_list = QListWidget()
        layout.addWidget(self.picked_points_list)

        dock.setWidget(widget)

        self.picked_points = []

        def callback(point):
            """鼠标拾取点的回调函数"""
            self.picked_points.append(point)
            self.picked_points_list.addItem(f"Point: {point}")

            self.plotter_widget.add_mesh(pv.Sphere(center=point, radius=0.1), color="red")

        self.plotter_widget.enable_point_picking(callback=callback, show_message=True, color="red", point_size=10)

        def finish_selection():
            """完成点集合选择"""
            if self.picked_points:
                selected_item = self.tree_widget.currentItem()
                if selected_item:
                    new_item = QTreeWidgetItem(selected_item)
                    new_item.setText(0, f"Point Set ({len(self.picked_points)} points)")
                    selected_item.addChild(new_item)

                QMessageBox.information(self, "Selection Completed", f"Selected {len(self.picked_points)} points.")

                self.picked_points = []
                self.picked_points_list.clear()
                self.plotter_widget.disable_picking()
                dock.close()
            else:
                QMessageBox.warning(self, "No Points", "No points were selected.")

        finish_button.clicked.connect(finish_selection)


    def voxelize_model(self):
        # Use a executable file to voxelize the model
        pass

    def reset_camera(self):
        self.plotter_widget.reset_camera()

    def set_camera_position(self, view):

        if view == "front":
            self.plotter_widget.camera_position = 'xy'
        elif view == "back":
            self.plotter_widget.camera_position = 'yx'
        elif view == "left":
            self.plotter_widget.camera_position = 'zx'
        elif view == "right":
            self.plotter_widget.camera_position = 'xz'
        elif view == "top":
            self.plotter_widget.camera_position = 'yz'
        elif view == "bottom":
            self.plotter_widget.camera_position = 'zy'
        else:
            print("Invalid view option")

        # 更新相机位置
        self.plotter_widget.reset_camera()

    def update_opacity(self):
        selected_item = self.tree_widget.currentItem()
        if not selected_item:
            QMessageBox.warning(self, "No Selection", "Please select a model to update opacity.")
            return

        file_name = selected_item.text(0)
        if file_name in self.models:
            mesh, _, actor = self.models[file_name]

            opacity_value = self.opacity_slider.value() / 100.0 # 0.0 to 1.0
            actor.GetProperty().SetOpacity(opacity_value)
            self.plotter_widget.update()
        else:
            QMessageBox.warning(self, "Error", "The selected model is not valid.")

    def update_colormap(self):
        selected_item = self.tree_widget.currentItem()
        if not selected_item:
            QMessageBox.warning(self, "No Selection", "Please select a model to update colormap.")
            return

        file_name = selected_item.text(0)
        if file_name in self.models:
            mesh, _, actor = self.models[file_name]

            colormap_name = self.colormap_combo.currentText()
            actor.mapper.SetLookupTable(pv.LookupTable(colormap_name))
            self.plotter_widget.update()
        else:
            QMessageBox.warning(self, "Error", "The selected model is not valid.")

    def update_threshold(self):
        selected_item = self.tree_widget.currentItem()
        if not selected_item:
            QMessageBox.warning(self, "No Selection", "Please select a model to update threshold.")
            return

        file_name = selected_item.text(0)
        if file_name in self.models:
            mesh, _, actor = self.models[file_name]

            try:
                min_value = float(self.min_threshold.text())
                max_value = float(self.max_threshold.text()) 

                thresholded_mesh = mesh.threshold([min_value, max_value])
                self.plotter_widget.remove_actor(actor)


                new_actor = self.plotter_widget.add_mesh(thresholded_mesh, name=file_name, show_edges=True)
                self.models[file_name] = (thresholded_mesh, selected_item, new_actor)
                self.plotter_widget.update()
                self.update_display_mode()
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "Please enter valid numbers for the threshold.")
        else:
            QMessageBox.warning(self, "Error", "The selected model is not valid.")

    def on_tree_item_changed(self, item, column):
        file_name = item.text(0)
        if file_name in self.models:
            _, _, actor = self.models[file_name]
            if item.checkState(0) == Qt.Checked:
                actor.VisibilityOn()
            else:
                actor.VisibilityOff()
            self.plotter_widget.update()


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
