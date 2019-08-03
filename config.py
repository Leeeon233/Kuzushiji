import os

# dataset
DATA_ROOT = '/disk2/zhaoliang/datasets/Kuzushiji'
TRAIN_CSV = os.path.join(DATA_ROOT, 'train.csv')
TRAIN_IMAGES = os.path.join(DATA_ROOT, 'train_images')
CROP_TRAIN_IMAGES = os.path.join(DATA_ROOT, 'crop_train_images')
TEST_IMAGES = os.path.join(DATA_ROOT, 'test_images')
CROP_TEST_IMAGES = os.path.join(DATA_ROOT, 'crop_test_images')
SAVE_ALL_CHARS = os.path.join(DATA_ROOT, 'all_crop_chars')
MAP_CSV = os.path.join(DATA_ROOT, 'unicode_translation.csv')
TRAIN_FILE = os.path.join(DATA_ROOT, 'train.txt')
VAL_FILE = os.path.join(DATA_ROOT, 'val.txt')
CHARS_FREQ_FILE = os.path.join(DATA_ROOT, "chars_freq.csv")
CHARS_DICT_FILE = os.path.join(DATA_ROOT, "chars_dict.csv")   # 裁剪出的全部字符图片与对应label
PROJECT_ROOT = '/disk2/zhaoliang/projects/Kuzushiji'
LOG_ROOT = os.path.join(PROJECT_ROOT, 'logs')

CROP_IMAGE_SIZE = 1280
CROP_IMAGE_OVERLAP = 0.7
INPUT_IMAGE_SIZE = 608

SEED = 12
# detection
VIS_SAVE_ROOT = '/disk2/zhaoliang/projects/Kuzushiji/perform'

# classification
CLS_RESIZE = 64
ERROR_CHAR = 'x'
BLUR_SIZE = 5
EPOCHES = 10000
BATCH_SIZE = 512
NUM_CLASSES = 500
NUM_PER_CLASS = 100