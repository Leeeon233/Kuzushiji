import os

# dataset
DATA_ROOT = '/disk2/zhaoliang/datasets/Kuzushiji'
TRAIN_CSV = os.path.join(DATA_ROOT, 'train.csv')
TRAIN_IMAGES = os.path.join(DATA_ROOT, 'train_images')
CROP_TRAIN_IMAGES = os.path.join(DATA_ROOT, 'crop_train_images')
TEST_IMAGES = os.path.join(DATA_ROOT, 'test_images')
CROP_TEST_IMAGES = os.path.join(DATA_ROOT, 'crop_test_images')
MAP_CSV = os.path.join(DATA_ROOT, 'unicode_translation.csv')
TRAIN_FILE = os.path.join(DATA_ROOT, 'train.txt')
VAL_FILE = os.path.join(DATA_ROOT, 'val.txt')

CROP_IMAGE_SIZE = 1280
CROP_IMAGE_OVERLAP = 0.7
INPUT_IMAGE_SIZE = 608

# detection
VIS_SAVE_ROOT = '/disk2/zhaoliang/projects/Kuzushiji/perform'