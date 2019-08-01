import os
import pandas as pd
import numpy as np
import config as C
import cv2
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

df_train = pd.read_csv(C.TRAIN_CSV).dropna()
unicode_map = {codepoint: char for codepoint, char in pd.read_csv(C.MAP_CSV).values}
VIS = False


def resize_img_box(img, labels):
    scale = C.INPUT_IMAGE_SIZE / C.CROP_IMAGE_SIZE
    resize_img = cv2.resize(img, (C.INPUT_IMAGE_SIZE, C.INPUT_IMAGE_SIZE))
    labels = labels * scale
    return resize_img, labels


def crop_ori_img(img_path, labels):
    img_size = C.CROP_IMAGE_SIZE
    over_lap = C.CROP_IMAGE_OVERLAP
    img_name = os.path.basename(img_path)  # xxx.jpg
    print(img_name, img_name[:-4])
    img = cv2.imread(img_path)
    n = 0
    h, w, c = img.shape
    cur_y = 0
    flag_y = False
    while cur_y + img_size <= h:
        cur_x = 0
        flag_x = False
        while cur_x + img_size <= w:
            crop_img = img[cur_y: cur_y + img_size, cur_x: cur_x + img_size]
            # TODO
            crop_labels = get_croped_label(cur_x, cur_y, labels)
            if len(crop_labels) == 0:
                if flag_x:
                    break
                cur_x += int(img_size * (1 - over_lap))
                if cur_x + img_size > w:
                    cur_x = w - img_size
                    flag_x = True
                continue

            crop_img, crop_labels = resize_img_box(crop_img, crop_labels)
            if VIS:
                visualize_training_data(crop_img, crop_labels, n)
            save_img_label(crop_img, crop_labels, f'{img_name[:-4]}_{n}')
            n += 1
            #
            if flag_x:
                break
            cur_x += int(img_size * (1 - over_lap))
            if cur_x + img_size > w:
                cur_x = w - img_size
                flag_x = True
        if flag_y:
            break
        cur_y += int(img_size * (1 - over_lap))
        if cur_y + img_size > h:
            cur_y = h - img_size
            flag_y = True


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def save_img_label(img, label, file_name):
    h, w, c = img.shape
    cv2.imwrite(os.path.join(C.CROP_TRAIN_IMAGES, file_name + '.jpg'), img)

    with open(os.path.join(C.CROP_TRAIN_IMAGES, f'{file_name}.txt'), 'w') as f:
        for box in label:
            x_min, x_max, y_min, y_max = box[0],  box[0] + box[2], box[1], box[1] + box[3]
            bb = convert((w, h), [x_min, x_max, y_min, y_max])
            f.write("0" + " " + " ".join([str(a) for a in bb]) + '\n')

    with open(C.TRAIN_FILE, 'a') as train_file:
        with open(C.VAL_FILE, 'a') as val_file:
            train_file.write(os.path.join(C.CROP_TRAIN_IMAGES, file_name + '.jpg') + '\n')
            # if np.random.randint(0, 9) < 1:
            #     val_file.write(os.path.join(C.CROP_TRAIN_IMAGES, file_name + '.jpg') + '\n')
            # else:
            #     train_file.write(os.path.join(C.CROP_TRAIN_IMAGES, file_name + '.jpg') + '\n')


def get_croped_label(x, y, labels):
    result = []

    for box in labels:
        x_min, y_min, w, h = box
        x_min = int(x_min)
        y_min = int(y_min)
        w = int(w)
        h = int(h)
        if x <= x_min and y <= y_min and x + C.CROP_IMAGE_SIZE >= x_min + w and y + C.CROP_IMAGE_SIZE >= y_min + h:
            result.append([x_min - x, y_min - y, w, h])
    return np.array(result)


def crop_make_label(img_name, labels):
    img_path = os.path.join(C.TRAIN_IMAGES, f'{img_name}.jpg')
    labels = np.array(labels.split(' ')).reshape(-1, 5)
    labels = labels[:, 1:]
    # codepoint, x_min, y_min, w, h = labels
    crop_ori_img(img_path, labels)


def main():
    for idx in tqdm(range(len(df_train))):
        img_name, labels = df_train.values[idx]
        crop_make_label(img_name, labels)


def visualize_training_data(image_fn, labels, n):
    # Convert annotation string to array

    # Read image
    imsource = Image.fromarray(image_fn).convert('RGBA')
    bbox_canvas = Image.new('RGBA', imsource.size)
    char_canvas = Image.new('RGBA', imsource.size)
    bbox_draw = ImageDraw.Draw(
        bbox_canvas)  # Separate canvases for boxes and chars so a box doesn't cut off a character
    char_draw = ImageDraw.Draw(char_canvas)

    for x, y, w, h in labels:
        x, y, w, h = int(x), int(y), int(w), int(h)
        # Convert codepoint to actual unicode character

        # Draw bounding box around character, and unicode character next to it
        bbox_draw.rectangle((x, y, x + w, y + h), fill=(255, 0, 255, 0), outline=(255, 0, 0, 255))
        # char_draw.text((x + w + fontsize / 4, y + h / 2 - fontsize), char, fill=(0, 0, 255, 255), font=font)

    imsource = Image.alpha_composite(Image.alpha_composite(imsource, bbox_canvas), char_canvas)
    imsource = imsource.convert("RGB")  # Remove alpha for saving in jpg format.
    plt.figure(figsize=(15, 15))
    plt.imshow(np.asarray(imsource), interpolation='lanczos')
    plt.savefig(f'debug/{n}.png')


if __name__ == '__main__':
    main()
