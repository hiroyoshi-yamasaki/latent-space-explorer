

class Model:
    def __init__(self):
        self.compressed = None

    def fit(self):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def generate(self, components):
        raise NotImplementedError

