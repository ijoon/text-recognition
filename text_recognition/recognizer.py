

from config import cfg
from custom_models import vgg_lstm


class Recognizer:
    def __init__(self, cfg):
        self.cfg = cfg

        if cfg['model'] == 'VGGLSTM':
            self.model = vgg_lstm.VGGLSTM()
        else:
            self.model = vgg_lstm.VGGLSTM()

    def train(self):
        self.model.train()

    def predict(self):
        output_value = self.model.predict() # text result


if __name__ == '__main__':
    pass
