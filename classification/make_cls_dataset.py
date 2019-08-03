from PIL import Image
from tqdm import tqdm
import pandas as pd
import config as C
import numpy as np
import os
import cv2

np.random.seed(C.SEED)


def resize_square_img(img):
    """
    改成正方形
    :param img:
    :return:
    """
    h = img.shape[0]
    w = img.shape[1]
    value = (0, 0)
    if h > w:
        le = int((h - w) / 2)
        r = h - w - le
        border = [0, 0, le, r]
    else:
        t = int((w - h) / 2)
        b = w - h - t
        border = [t, b, 0, 0]
    img = cv2.copyMakeBorder(img, border[0], border[1], border[2], border[3], cv2.BORDER_CONSTANT, value=value)
    # print(img.shape)
    return img


def threshold(img):
    """
    二值化
    :param img:
    :return:
    """
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    img = cv2.medianBlur(img, C.BLUR_SIZE)
    # print("阈值 ", ret)
    return img


df_char = pd.read_csv(C.CHARS_FREQ_FILE)


def crop_all_char(image_name, labels):
    """
    将图片中的全部文字剪裁出来保存到csv中, label 0 是少见类, 所以label从1开始
    :param image_name:
    :param labels:
    :return:
    """
    img_path = os.path.join(C.TRAIN_IMAGES, f'{image_name}.jpg')
    labels = np.array(labels.split(' ')).reshape(-1, 5)
    img = cv2.imread(img_path)
    is_train = 1
    n = 0
    if np.random.randint(0, 10) < 1:
        is_train = 0
    for idx, (codepoint, x_min, y_min, w, h) in enumerate(labels):

        crop_img_name = f'{image_name}_{idx}'
        crop_img_path = os.path.join(C.SAVE_ALL_CHARS, f'{crop_img_name}.jpg')
        if os.path.exists(crop_img_path):
            continue
        n += 1
        x_min, y_min, w, h = int(x_min), int(y_min), int(w), int(h)
        label_index = df_char[df_char['char'] == codepoint]['idx'].values[0]
        jp_char = df_char[df_char['char'] == codepoint]['jp_char'].values[0]
        crop_img = img[y_min:y_min + h, x_min:x_min + w]
        crop_img = threshold(crop_img)
        crop_img = resize_square_img(crop_img)

        cv2.imwrite(crop_img_path, crop_img)
        save_content = {
            'img_name': crop_img_name,
            'label': label_index + 1,
            'jp_char': jp_char,
            'is_train': is_train
        }
        # save_content = [crop_img_name, label_index+1, jp_char, is_train]
        char_dict = pd.DataFrame(save_content, index=[0])  # , columns=['img_name', 'label', 'jp_char', 'is_train'])
        char_dict.to_csv(C.CHARS_DICT_FILE, mode='a', header=None, index=False)
    return n


def main():
    df_train = pd.read_csv(C.TRAIN_CSV).dropna()
    sum = len(os.listdir(C.SAVE_ALL_CHARS))
    for idx in tqdm(range(len(df_train))):  #
        img_name, labels = df_train.values[idx]
        sum += crop_all_char(img_name, labels)
        print(sum)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
