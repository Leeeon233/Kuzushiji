import numpy as np
import config as C


def crop_ori_img_scale(img):
    OVER_LAP = C.CROP_IMAGE_OVERLAP
    h, w, c = img.shape
    IMG_SIZE = C.CROP_IMAGE_SIZE
    stride = int(IMG_SIZE * (1 - OVER_LAP))
    sub_img = []
    left_top_points = []
    count_h = int(np.ceil(h / stride))
    count_w = int(np.ceil(w / stride))
    for idx_h in range(count_h):
        y_min = idx_h * stride if idx_h + 1 != count_h else h - IMG_SIZE
        y_max = y_min + IMG_SIZE
        for idx_w in range(count_w):
            x_min = idx_w * stride if idx_w + 1 != count_w else w - IMG_SIZE
            x_max = x_min + IMG_SIZE
            sub_img.append(img[y_min:y_max, x_min:x_max])
            left_top_points.append((x_min, y_min))
    return sub_img, left_top_points


def merge_sub_bbox(boxes_s, scores_s, points):
    assert len(boxes_s) == len(points)
    assert len(boxes_s) == len(scores_s)
    pred_boxes = np.zeros((0, 4))
    scores_ = np.zeros((0,))

    for boxes, point, scores in zip(boxes_s, points, scores_s):
        boxes = np.array(boxes)
        if len(boxes) == 0:
            continue
        _point = np.array([point[0], point[1], point[0], point[1]])
        boxes[:, :4] += _point

        pred_boxes = np.concatenate([pred_boxes, boxes])
        scores_ = np.concatenate([scores_, scores])
    return pred_boxes, scores_
