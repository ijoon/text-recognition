CHAR_VECTOR = '0123456789 '

letters = [letter for letter in CHAR_VECTOR]

CLASS_NUM = len(letters) + 1

IMG_W, IMG_H = 128, 64

# Network parameters
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 16

DOWNSAMPLE_FACTOR = 4
MAX_TEXT_LEN = 2
