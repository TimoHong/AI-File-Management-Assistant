from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, 
                            QLabel, QSpinBox, QDialogButtonBox)
from PyQt5.QtCore import Qt

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('AI Grouping Settings')
        layout = QVBoxLayout(self)

        # Number of groups setting
        group_layout = QHBoxLayout()
        group_layout.addWidget(QLabel('Number of groups:'))
        self.group_spin = QSpinBox()
        self.group_spin.setRange(2, 10)
        self.group_spin.setValue(3)
        group_layout.addWidget(self.group_spin)
        layout.addLayout(group_layout)

        # OK/Cancel buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_settings(self):
        return {
            'num_groups': self.group_spin.value()
        }