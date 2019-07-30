import numpy as np
import copy
from utils import cal_edge_inner, merge_box


class MergeTool:
    def __init__(self, class_num=10, momentum=0.5, edge_alpha=0, score_threshold=0):
        # self.momentum = momentum
        self.edge_alpha = edge_alpha
        self.class_num = class_num
        self.score_threshold = score_threshold

    def _classes_merge(self, bboxes, vertical=True):

        momentum_num = np.ones(len(bboxes))  # 动量系数包含越多的框，越可能吞并临近的框 如果为0表示已经被吞并
        # 按score降序排列
        bboxes = bboxes[np.argsort(-bboxes[:, 5])]
        # 遍历全部box
        for cur_idx in range(len(bboxes)):
            # 此框被保留
            if momentum_num[cur_idx] > 0:
                cur_x_min, cur_y_min, cur_x_max, cur_y_max, cur_class_name, cur_score = bboxes[cur_idx]

                # 按中心与当前框距离越近排序
                # vertical：
                compare_boxes = copy.deepcopy(bboxes)
                compare_idxs = np.argsort(np.abs(compare_boxes[:, 3] + compare_boxes[:, 1]) - (cur_y_max + cur_y_min))[
                               ::-1]
                for compare_idx in compare_idxs:

                    if momentum_num[compare_idx] > 0 and cur_idx != compare_idx:
                        (compare_x_min,
                         compare_y_min,
                         compare_x_max,
                         compare_y_max, _,
                         compare_score) = bboxes[compare_idx]
                        # 如果相交
                        if cal_edge_inner((cur_x_min, cur_x_max),
                                          (compare_x_min, compare_x_max),
                                          is_momentum=False):
                            if cal_edge_inner((cur_y_min, cur_y_max),
                                              (compare_y_min, compare_y_max),
                                              momentum_num=max(momentum_num[cur_idx], momentum_num[compare_idx]),
                                              alpha=self.edge_alpha):
                                bboxes[cur_idx] = merge_box(bboxes[cur_idx],
                                                            bboxes[compare_idx],
                                                            max(cur_score, compare_score))
                                momentum_num[cur_idx] += 1
                                momentum_num[compare_idx] = 0

                compare_boxes = copy.deepcopy(bboxes)
                compare_idxs = np.argsort(np.abs(compare_boxes[:, 2] + compare_boxes[:, 0])
                                          - (cur_x_max + cur_x_min))[::-1]
                momentum_num[momentum_num > 0] = 1
                for compare_idx in compare_idxs:
                    if momentum_num[compare_idx] > 0 and cur_idx != compare_idx:
                        (compare_x_min,
                         compare_y_min,
                         compare_x_max,
                         compare_y_max, _,
                         compare_score) = bboxes[compare_idx]
                        # 如果相交
                        if cal_edge_inner((cur_y_min, cur_y_max),
                                          (compare_y_min, compare_y_max),
                                          is_momentum=False):
                            if cal_edge_inner((cur_x_min, cur_x_max),
                                              (compare_x_min, compare_x_max),
                                              momentum_num=momentum_num[cur_idx],
                                              alpha=self.edge_alpha):
                                bboxes[cur_idx] = merge_box(bboxes[cur_idx],
                                                            bboxes[compare_idx],
                                                            max(cur_score, compare_score))
                                momentum_num[cur_idx] += 1
                                momentum_num[compare_idx] = 0

        return bboxes[momentum_num > 0]

    def merge(self, bboxes):
        """

        :param bboxes  np.array([[x_min, y_min, x_max, y_max, class_name, score], ...])
        :return:
        """

        if isinstance(bboxes, list):
            bboxes = np.array(bboxes)

        box_num, dim = bboxes.shape
        assert 6 == dim, "dim must 6"
        result = np.empty((0, 6))
        # remove score smaller than threshold
        bboxes = bboxes[bboxes[:, 5] > self.score_threshold]
        class_names = bboxes[:, 4]
        for class_name in range(self.class_num):
            idxs = (class_names == class_name)
            boxes_by_class = bboxes[idxs]
            if len(boxes_by_class) == 0:
                continue
            # x_mins = boxes_by_class[:, 0]
            # y_mins = boxes_by_class[:, 1]
            # x_maxs = boxes_by_class[:, 2]
            # y_maxs = boxes_by_class[:, 3]
            # print(class_name)
            # print(boxes_by_class)
            result = np.concatenate([result, self._classes_merge(boxes_by_class)])

        return result


if __name__ == '__main__':
    boxes = [[0, 0, 100, 10, 1, 1],
             [0, 90, 100, 110, 1, 2],
             [0, 15, 100, 115, 1, 3],
             [0, 11, 100, 120, 1, 4],
             [0, 0, 100, 100, 3, 5],
             [0, 0, 100, 100, 2, 6],
             [0, 0, 100, 100, 3, 7],
             [0, 13, 110, 100, 1, 8]]
    print(MergeTool().merge(boxes))
