import numpy as np
import config as C
import cv2


def crop_ori_img_scale(img):
    img_size = C.INPUT_IMAGE_SIZE
    over_lap = 0.3
    h, w, c = img.shape
    cur_y = 0
    flag_y = False
    sub_img = []
    left_top_points = []

    while cur_y + img_size <= h:
        cur_x = 0
        flag_x = False
        while cur_x + img_size <= w:
            crop_img = img[cur_y: cur_y + img_size, cur_x: cur_x + img_size]
            # crop_img = cv2.resize(crop_img, (C.INPUT_IMAGE_SIZE, C.INPUT_IMAGE_SIZE))
            sub_img.append(crop_img)
            left_top_points.append([cur_x, cur_y])
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
        _point = np.array([point[0], point[1], 0, 0])
        boxes[:, :4] += _point

        pred_boxes = np.concatenate([pred_boxes, boxes])
        scores_ = np.concatenate([scores_, scores])
    return pred_boxes, scores_


def non_max_suppression_fast(boxes, scores, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]-boxes[:, 2]/2
    y1 = boxes[:, 1]-boxes[:, 3]/2
    x2 = boxes[:, 0]+boxes[:, 2]/2
    y2 = boxes[:, 1]+boxes[:, 3]/2
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the score of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        # 计算重叠区域的左上与右下坐标
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        # 计算重叠区域的长宽
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        # 计算重叠区域占原区域的面积比（重叠率）
        overlap = (w * h) / area[idxs[:last]]

        # 删除所有重叠率大于阈值的边界框
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    if len(pick) == 0:
        return [], []
    return boxes[pick], scores[pick]


def nms(bounding_boxes, confidence_score, threshold):
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []

    # Bounding boxes
    boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    return picked_boxes, picked_score
