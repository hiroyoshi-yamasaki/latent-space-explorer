import numpy as np
from matplotlib import pyplot as plt
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QSlider, QPushButton, QGridLayout, QRadioButton
from PyQt5.QtGui import QImage, QPixmap
from models.model import Model
from utils import apply_cmap
from params import *

SLIDER_MAX = 100

# Basic layout #################################################################################################
# ----------------------     ----------------------   image is shown on the left frame                         #
# |                    |     |                    |   sample button for resampling on top right                #
# |                    |     |    sample_button   |   sliders for controlling latent parameters shown on       #
# |                    |     |                    |   bottom right                                             #
# |      display       |     |--------------------|                                                            #
# |                    |     |                    |                                                            #
# |                    |     |       sliders      |                                                            #
# |                    |     |                    |                                                            #
# ----------------------     ----------------------                                                            #
# ##############################################################################################################


def _reformat_image(image):
    # Apply color map to grayscale image and draw
    cmap = plt.get_cmap(CMAP)
    image = (image * 255).astype(np.uint8)
    image = apply_cmap(image, cmap)
    return image.reshape(IMG_SIZE, IMG_SIZE, 3)


class Window(QWidget):
    def __init__(self, title: str, parent=None) -> None:
        super(Window, self).__init__(parent)

        # Main attributes
        self.model = Model()

        # Main window
        self.setWindowTitle(title)
        self.resize(WIDTH, HEIGHT)

        # Set up display area
        self.display_label = QLabel()
        self._init_display()

        # Set up top right control panel
        self.sample_button = QPushButton("Sample")
        self.resize(BUTTON_WIDTH, BUTTON_HEIGHT)
        self.sample_button.clicked.connect(self._sample)

        # Set up slider grid
        self.slider_grid = QGridLayout()
        self.sliders = []
        for row in range(N_ROWS):
            for col in range(N_COLS):
                slider = QSlider(Qt.Vertical)
                slider.setMinimum(0)
                slider.setMaximum(SLIDER_MAX)
                slider.valueChanged.connect(self._update)
                self.slider_grid.addWidget(slider, row, col)
                self.sliders.append(slider)

        # Left (display) vs right (control panel)
        self.h_layout = QHBoxLayout()
        self.setLayout(self.h_layout)
        self.h_layout.addWidget(self.display_label)

        # Top (sample button) vs bottom (sliders)
        self.v_layout = QVBoxLayout()
        self.v_layout.addWidget(self.sample_button)
        self.v_layout.addLayout(self.slider_grid)
        self.h_layout.addLayout(self.v_layout)

    def _init_display(self):
        self.display_label.setAlignment(Qt.AlignCenter)
        blank = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        self._draw(blank)

    def _sample(self):
        reconstruction, components = self.model.sample()

        # Apply color map to grayscale image and draw
        image = _reformat_image(reconstruction)
        self._draw(image)

        # Update values
        for i in range(N_ROWS * N_COLS):
            self.sliders[i].setValue(components[i] * SLIDER_MAX)

    def _update(self):

        components = []
        for slider in self.sliders:
            components.append(slider.value())
        image = self.model.generate(np.array(components))
        image = _reformat_image(image)
        self._draw(image)

    def _draw(self, image):
        qimage = QImage(image, image.shape[1], image.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap(qimage)
        pixmap = pixmap.scaled(DISPLAY, DISPLAY, Qt.KeepAspectRatio)
        self.display_label.setPixmap(pixmap)
