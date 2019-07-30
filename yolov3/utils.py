import cv2 as cv
import numpy as np


def plot_img_bbox(img, bbox, color):
    for b in bbox:
        cv.rectangle(img, (b[0], b[1]), (b[2], b[3]), color, thickness=3)
        cv.putText(img, b[4], (b[0], b[1]), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0),
                   thickness=2)
    cv.imshow('result', img)
    k = cv.waitKey(0)


def mat_inter(box1, box2):
    # 判断两个矩形是否相交
    # box=(xA,yA,xB,yB)
    x01, y01, x02, y02 = box1
    x11, y11, x12, y12 = box2

    lx = abs((x01 + x02) / 2 - (x11 + x12) / 2)
    ly = abs((y01 + y02) / 2 - (y11 + y12) / 2)
    sax = abs(x01 - x02)
    sbx = abs(x11 - x12)
    say = abs(y01 - y02)
    sby = abs(y11 - y12)
    if lx <= (sax + sbx) / 2 and ly <= (say + sby) / 2:
        return True
    else:
        return False


def cal_edge_iou(edge_a, edge_b):
    """
    :param edge_a:  [start, end]
    :param edge_b:
    :return:
    """
    if cal_edge_inner(edge_a, edge_b):
        points = sorted(list(edge_a) + list(edge_b))
        return (points[2] - points[1]) / (points[3] - points[0])
    else:
        return 0


def cal_edge_inner(edge_a, edge_b, is_momentum=True, momentum_num=None, alpha=0.25):
    if is_momentum:
        assert momentum_num is not None, "if is_momentum is True, momentum_num can not None"
        momentum_alpha = 1 / (1 + np.exp(-momentum_num)) * alpha
        length = max((edge_a[1] - edge_a[0]),(edge_b[1]-edge_b[0]))
        edge_a = [edge_a[0] - length * momentum_alpha, edge_a[1] + length * momentum_alpha]
        # [np.mean(edge_a) - momentum_alpha * (edge_a[1] - edge_a[0]),
        # np.mean(edge_a) + momentum_alpha * (edge_a[1] - edge_a[0])]
        # edge_b = [np.mean(edge_b) - momentum_alpha * (edge_b[1] - edge_b[0]),
        # np.mean(edge_b) + momentum_alpha * (edge_b[1] - edge_b[0])]
    return edge_a[0] <= edge_b[0] <= edge_a[1] or edge_b[0] <= edge_a[0] <= edge_b[1]  # TODO add =


def merge_box(box_a, box_b, score):
    x_min = min(box_a[0], box_b[0])
    y_min = min(box_a[1], box_b[1])
    x_max = max(box_a[2], box_b[2])
    y_max = max(box_a[3], box_b[3])
    return np.array([x_min, y_min, x_max, y_max, box_a[4], score])
