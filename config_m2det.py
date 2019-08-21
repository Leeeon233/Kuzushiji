import os

# dataset
DATA_ROOT = '/shared_disk/leonzhao/datasets/ku'
TRAIN_CSV = os.path.join(DATA_ROOT, 'train.csv')
TRAIN_IMAGES = os.path.join(DATA_ROOT, 'train_images')
TEST_IMAGES = os.path.join(DATA_ROOT, 'test_images')

MAP_CSV = os.path.join(DATA_ROOT, 'unicode_translation.csv')
TRAIN_FILE = os.path.join(DATA_ROOT, 'train.txt')      # 自行划分训练集文件
VAL_FILE = os.path.join(DATA_ROOT, 'val.txt')          # 自行划分测试集文件

SUBMIT_ROOT = os.path.join(DATA_ROOT, 'submits')
if not os.path.exists(SUBMIT_ROOT):
    os.makedirs(SUBMIT_ROOT)
SUBMIT_FILE = os.path.join(DATA_ROOT, "submit.csv")

PROJECT_ROOT = '/home/zhaoliang/project/Kuzushiji'
LOG_ROOT = os.path.join(PROJECT_ROOT, 'logs')

CROP_IMAGE_SIZE = 1280
CROP_IMAGE_OVERLAP = 0.3
INPUT_IMAGE_SIZE = 800

SEED = 12
# detection

CROP_TRAIN_IMAGES = os.path.join(DATA_ROOT, 'crop_train_images')   # 步长剪裁后训练集
CROP_TRAIN_IMAGES_IMAGE = os.path.join(CROP_TRAIN_IMAGES, 'images')   # 训练集图片
CROP_TRAIN_IMAGES_LABEL = os.path.join(CROP_TRAIN_IMAGES, 'labels')   # 训练集标签
if not os.path.exists(CROP_TRAIN_IMAGES_IMAGE):
    os.makedirs(CROP_TRAIN_IMAGES_IMAGE)
if not os.path.exists(CROP_TRAIN_IMAGES_LABEL):
    os.makedirs(CROP_TRAIN_IMAGES_LABEL)
CROP_TEST_IMAGES = os.path.join(DATA_ROOT, 'crop_test_images')
if not os.path.exists(CROP_TEST_IMAGES):
    os.makedirs(CROP_TEST_IMAGES)
DETECTION_TRAIN_FILE = os.path.join(DATA_ROOT, 'train_det.txt')
DETECTION_VAL_FILE = os.path.join(DATA_ROOT, 'val_det.txt')
VIS_SAVE_ROOT = os.path.join(PROJECT_ROOT, 'perform')
# LIB_SO_PATH = '/disk2/zhaoliang/projects/Kuzushiji/yolov3/libdarknet.so'

# classification
CHARS_FREQ_FILE = os.path.join(DATA_ROOT, "chars_freq.csv")
CHARS_DICT_FILE = os.path.join(DATA_ROOT, "chars_dict.csv")   # 裁剪出的全部字符图片与对应label
SAVE_ALL_CHARS = os.path.join(DATA_ROOT, 'all_crop_chars')    # 剪裁后全部图片
CLS_RESIZE = 64
ERROR_CHAR = 'x'
BLUR_SIZE = 5
EPOCHES = 10000
BATCH_SIZE = 512
NUM_CLASSES = 1000
NUM_PER_CLASS = 200