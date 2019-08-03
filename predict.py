import os

import cv2
import numpy as np
import pandas as pd

import config as C
from classification.classifier import Classifier
from yolov3.darknet import Detector

os.environ["CUDA_VISIBLE_DEVICES"] = '3'


class Predictor:
    def __init__(self):
        self.detector = Detector(
            thresh=0.85,
            weight_path='/disk2/zhaoliang/projects/Kuzushiji/yolov3/backup_all/train_best.weights',
            config_path='/disk2/zhaoliang/projects/Kuzushiji/yolov3/kanji/test.cfg',
            meta_path='/disk2/zhaoliang/projects/Kuzushiji/yolov3/kanji/kanji.data'
        )
        self.classifier = Classifier(
            checkpoint='/disk2/zhaoliang/projects/Kuzushiji/classification/checkpoint500_5.pt'
        )
        self._init_model_data()

    def _init_model_data(self):
        self.dict = {}
        df_char = pd.read_csv(C.CHARS_FREQ_FILE)
        for idx, row in df_char.iterrows():
            index = int(row['idx']) + 1
            char = row['char']
            self.dict[index] = char
            if len(self.dict) == C.NUM_CLASSES:
                break
        if not os.path.exists(C.SUBMIT_FILE):
            pd.DataFrame(columns=['image_id', 'labels']).to_csv(C.SUBMIT_FILE, mode='w', index=False)

    def save2submit(self, image_id, label):
        # save_content = {
        #     'image_id': image_id,
        #     'labels': label
        # }
        # char_dict = pd.DataFrame(save_content, index=[0])
        # char_dict.to_csv(C.SUBMIT_FILE, mode='a', header=None, index=False)
        with open(os.path.join(C.SUBMIT_ROOT,f'{image_id}.txt'), 'w') as f:
            f.write(label)

    def _resize_square_img(self, img):
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
        img = cv2.resize(img, (C.CLS_RESIZE, C.CLS_RESIZE))
        return img

    def _threshold(self, img):
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

    def _crop_resize(self, img, boxes):
        crop_imgs = []
        centers = []
        for box in boxes:
            center_x = box[0]
            center_y = box[1]
            x = int(box[0] - box[2] / 2)
            y = int(box[1] - box[3] / 2)
            xs = int(box[2])
            ys = int(box[3])
            crop_img = img[y:y + ys, x:x + xs]
            if 0 in crop_img.shape:    # TODO  宽高存在-5
                continue
            crop_img = self._threshold(crop_img)
            crop_img = self._resize_square_img(crop_img)
            crop_img = np.array(crop_img).reshape((1, 64, 64))
            crop_imgs.append(crop_img)
            centers.append((center_x, center_y))
        return crop_imgs, centers

    def predict_detection(self, image_path):
        boxes = np.array(self.detector.detect_one_image(image_path))
        boxes = boxes * C.CROP_IMAGE_SIZE / C.INPUT_IMAGE_SIZE
        return boxes

    def _make_laebl(self, centers, labels):
        result = ''
        for label, center in zip(labels, centers):
            l = 'U+725B' if label == 0 else self.dict[label]
            result += f'{l} {int(center[0])} {int(center[1])} '
        return result[:-1]

    def predict_by_path(self, image_path):
        img = cv2.imread(image_path)
        boxes = self.predict_detection(image_path)
        if len(boxes) == 0:
            return ''
        crop_imgs, centers = self._crop_resize(img, boxes)
        if len(crop_imgs) > 100:
            tmp_label = []
            for i in range(len(crop_imgs)//100+1):
                crop_imgs_part = crop_imgs[i * 100: (i+1) * 100]
                if len(crop_imgs_part) == 0:
                    continue
                ll = self.classifier.predict(crop_imgs_part)
                tmp_label += list(ll)
            labels = np.array(tmp_label)
        else:
            labels = self.classifier.predict(crop_imgs)
        label = self._make_laebl(centers, labels)
        return label

    def predict_by_id(self, image_id):
        image_path = os.path.join(C.TEST_IMAGES, f'{image_id}.jpg')
        label = self.predict_by_path(image_path)
        self.save2submit(image_id, label)


if __name__ == '__main__':
    p = Predictor()
    # img_path = '/disk2/zhaoliang/datasets/Kuzushiji/train_images/hnsd006-026.jpg'
    # pred_string = p.predict_by_path(img_path)
    # import matplotlib.pyplot as plt
    # from PIL import Image, ImageDraw, ImageFont
    #
    # fontsize = 50
    # font = ImageFont.truetype('./NotoSansCJKjp-Regular.otf', fontsize, encoding='utf-8')
    # unicode_map = {codepoint: char for codepoint, char in pd.read_csv(C.MAP_CSV).values}
    #
    # def visualize_predictions(image_fn, labels):
    #     # Convert annotation string to array
    #     labels = np.array(labels.split(' ')).reshape(-1, 3)
    #
    #     # Read image
    #     imsource = Image.open(image_fn).convert('RGBA')
    #     bbox_canvas = Image.new('RGBA', imsource.size)
    #     char_canvas = Image.new('RGBA', imsource.size)
    #     bbox_draw = ImageDraw.Draw(
    #         bbox_canvas)  # Separate canvases for boxes and chars so a box doesn't cut off a character
    #     char_draw = ImageDraw.Draw(char_canvas)
    #
    #     for codepoint, x, y in labels:
    #         x, y = int(x), int(y)
    #         char = unicode_map[codepoint]  # Convert codepoint to actual unicode character
    #
    #         # Draw bounding box around character, and unicode character next to it
    #         bbox_draw.rectangle((x - 10, y - 10, x + 10, y + 10), fill=(255, 0, 0, 255))
    #         char_draw.text((x + 25, y - fontsize * (3 / 4)), char, fill=(255, 0, 0, 255), font=font)
    #
    #     imsource = Image.alpha_composite(Image.alpha_composite(imsource, bbox_canvas), char_canvas)
    #     imsource = imsource.convert("RGB")  # Remove alpha for saving in jpg format.
    #     return np.asarray(imsource)
    #
    #
    # # %%
    # viz = visualize_predictions(img_path, pred_string)
    # cv2.imwrite('predict.jpg', viz)
    #
    # plt.figure(figsize=(15, 15))
    # plt.imshow(viz, interpolation='lanczos')
    from tqdm import tqdm
    for image_name in tqdm(os.listdir(C.TEST_IMAGES)):
        image_id = image_name[:-4]
        if os.path.exists(os.path.join(C.SUBMIT_ROOT,f'{image_id}.txt')):
            continue
        print(image_id)
        p.predict_by_id(image_id)