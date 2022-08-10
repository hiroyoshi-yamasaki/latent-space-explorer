import sys
from PyQt5.QtWidgets import QApplication
from window import Window
from models.autoencoders import AutoEncoder


def main():

    app = QApplication(sys.argv)
    main_window = Window(title="Latent Space Explorer")

    # TODO: only temporary solution
    model = AutoEncoder()
    model.load("autoencoder.pt")
    main_window.model = model

    main_window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
