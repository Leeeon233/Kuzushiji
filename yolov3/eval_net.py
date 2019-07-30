#!python3
"""
Python 3 wrapper for identifying objects in images

Requires DLL compilation

Both the GPU and no-GPU version should be compiled; the no-GPU version should be renamed "yolo_cpp_dll_nogpu.dll".

On a GPU system, you can force CPU evaluation by any of:

- Set global variable DARKNET_FORCE_CPU to True
- Set environment variable CUDA_VISIBLE_DEVICES to -1
- Set environment variable "FORCE_CPU" to "true"


To use, either run performDetect() after import, or modify the end of this file.

See the docstring of performDetect() for parameters.

Directly viewing or returning bounding-boxed images requires scikit-image to be installed (`pip install scikit-image`)


Original *nix 2.7: https://github.com/pjreddie/darknet/blob/0f110834f4e18b30d5f101bf8f1724c34b7b83db/python/darknet.py
Windows Python 2.7 version: https://github.com/AlexeyAB/darknet/blob/fc496d52bf22a0bb257300d3c79be9cd80e722cb/build/darknet/x64/darknet.py

@author: Philip Kahn
@date: 20180503
"""
# pylint: disable=R, W0401, W0614, W0703
from ctypes import *
import math
import random
import os
import numpy as np
import time

from scipy.misc import imread
from skimage import io, draw


def sample(probs):
    s = sum(probs)
    probs = [a / s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs) - 1


def c_array(ctype, values):
    arr = (ctype * len(values))()
    arr[:] = values
    return arr


class BOX(Structure):
    _fields_ = [("x", c_float), ("y", c_float), ("w", c_float), ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [
        ("bbox", BOX),
        ("classes", c_int),
        ("prob", POINTER(c_float)),
        ("mask", POINTER(c_float)),
        ("objectness", c_float),
        ("sort_class", c_int),
    ]


class IMAGE(Structure):
    _fields_ = [("w", c_int), ("h", c_int), ("c", c_int), ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int), ("names", POINTER(c_char_p))]


# lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
# lib = CDLL("darknet.so", RTLD_GLOBAL)
hasGPU = True
if os.name == "nt":
    cwd = os.path.dirname(__file__)
    os.environ["PATH"] = cwd + ";" + os.environ["PATH"]
    winGPUdll = os.path.join(cwd, "yolo_cpp_dll.dll")
    winNoGPUdll = os.path.join(cwd, "yolo_cpp_dll_nogpu.dll")
    envKeys = list()
    for k, v in os.environ.items():
        envKeys.append(k)
    try:
        try:
            tmp = os.environ["FORCE_CPU"].lower()
            if tmp in ["1", "true", "yes", "on"]:
                raise ValueError("ForceCPU")
            else:
                print("Flag value '" + tmp + "' not forcing CPU mode")
        except KeyError:
            # We never set the flag
            if "CUDA_VISIBLE_DEVICES" in envKeys:
                if int(os.environ["CUDA_VISIBLE_DEVICES"]) < 0:
                    raise ValueError("ForceCPU")
            try:
                global DARKNET_FORCE_CPU
                if DARKNET_FORCE_CPU:
                    raise ValueError("ForceCPU")
            except NameError:
                pass
            # print(os.environ.keys())
            # print("FORCE_CPU flag undefined, proceeding with GPU")
        if not os.path.exists(winGPUdll):
            raise ValueError("NoDLL")
        lib = CDLL(winGPUdll, RTLD_GLOBAL)
    except (KeyError, ValueError):
        hasGPU = False
        if os.path.exists(winNoGPUdll):
            lib = CDLL(winNoGPUdll, RTLD_GLOBAL)
            print("Notice: CPU-only mode")
        else:
            # Try the other way, in case no_gpu was
            # compile but not renamed
            lib = CDLL(winGPUdll, RTLD_GLOBAL)
            print(
                "Environment variables indicated a CPU run, but we didn't find `"
                + winNoGPUdll
                + "`. Trying a GPU run anyway."
            )
else:
    lib = CDLL("./darknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

if hasGPU:
    set_gpu = lib.cuda_set_device
    set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [
    c_void_p,
    c_int,
    c_int,
    c_float,
    c_float,
    POINTER(c_int),
    c_int,
    POINTER(c_int),
    c_int,
]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

load_net_custom = lib.load_network_custom
load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
load_net_custom.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)


def array_to_image(arr):
    import numpy as np

    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2, 0, 1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w, h, c, data)
    return im, arr


def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        if altNames is None:
            nameTag = meta.names[i]
        else:
            nameTag = altNames[i]
        res.append((nameTag, out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res


netMain = None
metaMain = None
altNames = None


class YoloModel:
    def __init__(self, configPath, weightPath, metaPath, thersh=0.25):
        self.thresh = thersh
        self.hier_thresh = 0.5
        self.nms = 0.45
        self.crop_time = 0
        self.detect_time = 0
        self.netMain = None
        self.metaMain = None
        self.altNames = None
        self._init_model(configPath, weightPath, metaPath)

    def _init_model(self, config_path, weight_path, meta_path, batch_size=1):
        assert (
                0 < self.thresh < 1
        ), "Threshold should be a float between zero and one (non-inclusive)"
        if not os.path.exists(config_path):
            raise ValueError(
                "Invalid config path `" + os.path.abspath(config_path) + "`"
            )
        if not os.path.exists(weight_path):
            raise ValueError(
                "Invalid weight path `" + os.path.abspath(weight_path) + "`"
            )
        if not os.path.exists(meta_path):
            raise ValueError(
                "Invalid data file path `" + os.path.abspath(meta_path) + "`"
            )
        if self.netMain is None:
            self.netMain = load_net_custom(
                config_path.encode("ascii"), weight_path.encode("ascii"), 0, batch_size
            )  # batch size = 1
        if self.metaMain is None:
            self.metaMain = load_meta(meta_path.encode("ascii"))
        if self.altNames is None:
            # In Python 3, the metafile default access craps out on Windows (but not Linux)
            # Read the names file and create a list to feed to detect
            try:
                with open(meta_path) as metaFH:
                    meta_contents = metaFH.read()
                    import re

                    match = re.search(
                        "names *= *(.*)$", meta_contents, re.IGNORECASE | re.MULTILINE
                    )
                    if match:
                        result = match.group(1)
                    else:
                        result = None
                    try:
                        if os.path.exists(result):
                            with open(result) as namesFH:
                                names_list = namesFH.read().strip().split("\n")
                                self.altNames = [x.strip() for x in names_list]
                    except TypeError:
                        pass
            except Exception as e:
                print(e)
                pass
        print("Initialized detector")

    def crop_ori_img_scale(self, img):
        # IMG_SIZE = 1024
        OVER_LAP = 0.5
        h, w, c = img.shape
        IMG_SIZE = 300 #min(h, w) // 4
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

    def merge_sub_bbox(self, boxes_s, scores_s, points):
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

    def vis_detection(
            self, image, final_boxes, final_scores, save_path=None, show=False
    ):
        print(
            "*** " + str(len(final_scores)) + " Results, color coded by confidence ***"
        )
        imcaption = []
        for bounds, confidence in zip(final_boxes, final_scores):
            label = "bad"

            pstring = label + ": " + str(np.rint(100 * confidence)) + "%"
            imcaption.append(pstring)
            print(pstring)

            shape = image.shape
            # x = shape[1]
            # xExtent = int(x * bounds[2] / 100)
            # y = shape[0]
            # yExtent = int(y * bounds[3] / 100)
            yExtent = int(bounds[3])
            xEntent = int(bounds[2])
            # Coordinates are around the center
            xCoord = int(bounds[0] - bounds[2] / 2)
            yCoord = int(bounds[1] - bounds[3] / 2)
            boundingBox = [
                [xCoord, yCoord],
                [xCoord, yCoord + yExtent],
                [xCoord + xEntent, yCoord + yExtent],
                [xCoord + xEntent, yCoord],
            ]
            # Wiggle it around to make a 3px border
            rr, cc = draw.polygon_perimeter(
                [x[1] for x in boundingBox], [x[0] for x in boundingBox], shape=shape
            )
            rr2, cc2 = draw.polygon_perimeter(
                [x[1] + 1 for x in boundingBox],
                [x[0] for x in boundingBox],
                shape=shape,
            )
            rr3, cc3 = draw.polygon_perimeter(
                [x[1] - 1 for x in boundingBox],
                [x[0] for x in boundingBox],
                shape=shape,
            )
            rr4, cc4 = draw.polygon_perimeter(
                [x[1] for x in boundingBox],
                [x[0] + 1 for x in boundingBox],
                shape=shape,
            )
            rr5, cc5 = draw.polygon_perimeter(
                [x[1] for x in boundingBox],
                [x[0] - 1 for x in boundingBox],
                shape=shape,
            )
            boxColor = (
                int(255 * (1 - (confidence ** 2))),
                int(255 * (confidence ** 2)),
                0,
            )
            draw.set_color(image, (rr, cc), boxColor, alpha=0.8)
            draw.set_color(image, (rr2, cc2), boxColor, alpha=0.8)
            draw.set_color(image, (rr3, cc3), boxColor, alpha=0.8)
            draw.set_color(image, (rr4, cc4), boxColor, alpha=0.8)
            draw.set_color(image, (rr5, cc5), boxColor, alpha=0.8)
        if show:
            io.imshow(image)
            io.show()
        if save_path:
            result_path = os.path.join(
                "detect_result", "res_" + os.path.basename(save_path)
            )
            io.imsave(result_path, image)

    def detect(self, image_path, debug=False):
        if not os.path.exists(image_path):
            raise ValueError("Invalid image path `" + os.path.abspath(image_path) + "`")
        # detections = detect(self.netMain, self.metaMain, image_path.encode("ascii"), self.thresh)

        # def detect(self, netMain, metaMain, image, thresh=.5, hier_thresh=.5, nms=.45, debug=False):
        """
        Performs the meat of the detection
        """
        # pylint: disable= C0321
        # im = load_image(image_path.encode("ascii"), 0, 0)
        # import cv2
        # custom_image_bgr = cv2.imread(image) # use: detect(,,imagePath,)
        # custom_image = cv2.cvtColor(custom_image_bgr, cv2.COLOR_BGR2RGB)
        # custom_image = cv2.resize(custom_image,(lib.network_width(net), lib.network_height(net)), interpolation = cv2.INTER_LINEAR)
        # import scipy.misc
        # custom_image = scipy.misc.imread(image)
        # im, arr = array_to_image(custom_image)		# you should comment line below: free_image(im)

        img = imread(image_path)
        start = time.time()
        crop_imgs, points = self.crop_ori_img_scale(img)
        det_time = time.time()
        self.crop_time = det_time - start
        final_boxes = []
        final_scores = []
        final_points = []

        for im_in, point in zip(crop_imgs, points):
            boxes = []
            scores = []
            im, arr = array_to_image(im_in)

            if debug:
                print("Loaded image")
            num = c_int(0)
            if debug:
                print("Assigned num")
            pnum = pointer(num)
            if debug:
                print("Assigned pnum")
            predict_image(self.netMain, im)
            if debug:
                print("did prediction")
            # dets = get_network_boxes(net, custom_image_bgr.shape[1], custom_image_bgr.shape[0], thresh,
            # hier_thresh, None, 0, pnum, 0) # OpenCV
            dets = get_network_boxes(
                self.netMain,
                im.w,
                im.h,
                self.thresh,
                self.hier_thresh,
                None,
                0,
                pnum,
                0,
            )
            if debug:
                print("Got dets")
            num = pnum[0]
            if debug:
                print("got zeroth index of pnum")
            if self.nms:
                do_nms_sort(dets, num, self.metaMain.classes, self.nms)
            if debug:
                print("did sort")
            res = []
            if debug:
                print("about to range")
            for j in range(num):
                if debug:
                    print("Ranging on " + str(j) + " of " + str(num))
                if debug:
                    print(
                        "Classes: " + str(self.metaMain),
                        self.metaMain.classes,
                        self.metaMain.names,
                    )
                for i in range(self.metaMain.classes):
                    if debug:
                        print(
                            "Class-ranging on "
                            + str(i)
                            + " of "
                            + str(self.metaMain.classes)
                            + "= "
                            + str(dets[j].prob[i])
                        )
                    if dets[j].prob[i] > 0:
                        b = dets[j].bbox
                        if self.altNames is None:
                            nameTag = self.metaMain.names[i]
                        else:
                            nameTag = self.altNames[i]
                        if debug:
                            print("Got bbox", b)
                            print(nameTag)
                            print(dets[j].prob[i])
                            print((b.x, b.y, b.w, b.h))
                        res.append([nameTag, dets[j].prob[i], (b.x, b.y, b.w, b.h)])
                        boxes.append([b.x, b.y, b.w, b.h])
                        scores.append(dets[j].prob[i])

            # if debug: print("did range")
            # res = sorted(res, key=lambda x: -x[1])
            # return res
            final_scores.append(scores)
            final_boxes.append(boxes)
            final_points.append(point)
            # if debug:
            #     print("did sort")
            # free_image(im)
            # if debug:
            #     print("freed image")
            # free_detections(dets, num)
            # if debug:
            #     print("freed detections")
        det_toc = time.time()
        self.detect_time = det_toc - det_time

        final_boxes, final_scores = self.merge_sub_bbox(
            final_boxes, final_scores, final_points
        )
        im2show = np.copy(img)
        self.vis_detection(im2show, final_boxes, final_scores, image_path)


if __name__ == "__main__":
    config_path = "cloth/train.cfg"
    weight_path = "cloth/backup/train_3408.weights"
    meta_path = "cloth/voc.data"
    detector = YoloModel(config_path, weight_path, meta_path, 0.05)
    root = "/disk5/zhaoliang/phone_data_5.18/img"
    total_num = len(os.listdir(root))
    for index, filename in enumerate(os.listdir(root)):
        img_path = os.path.join(root, filename)
        detector.detect(img_path)
        print(
            f"{index + 1}/{total_num}\tcrop time : {detector.crop_time}\tdetect time: {detector.detect_time}"
        )
