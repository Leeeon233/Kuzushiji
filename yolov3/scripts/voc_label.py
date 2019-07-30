import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

sets = [("2007", "train"), ("2007", "test")]

classes = ["bad"]


def convert(size, box):
    dw = 1.0 / (size[0])
    dh = 1.0 / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(year, image_id):
    in_file = open(
        "/disk5/zhaoliang/VOC_Four_Cloth/Annotations/%s.xml" % ( image_id),
        encoding="utf-8",
    )
    out_file = open(
        "/disk5/zhaoliang/VOC_Four_Cloth/labels/%s.txt" % ( image_id), "w"
    )
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    for obj in root.iter("object"):
        difficult = obj.find("difficult").text
        cls = "bad"  # obj.find('name').text

        cls_id = classes.index(cls)
        xmlbox = obj.find("bndbox")
        b = (
            float(xmlbox.find("xmin").text),
            float(xmlbox.find("xmax").text),
            float(xmlbox.find("ymin").text),
            float(xmlbox.find("ymax").text),
        )
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + "\n")



for year, image_set in sets:
    if not os.path.exists("/disk5/zhaoliang/VOC_Four_Cloth/labels/"):
        os.makedirs("/disk5/zhaoliang/VOC_Four_Cloth/labels/")
    image_ids = (
        open(
            "/disk5/zhaoliang/VOC_Four_Cloth/ImageSets/Main/%s.txt"
            % (image_set)
        )
        .read()
        .strip()
        .split()
    )
    list_file = open("/disk5/zhaoliang/VOC_Four_Cloth/%s.txt" % (image_set), "w")
    for image_id in image_ids:
        list_file.write(
            "/disk5/zhaoliang/VOC_Four_Cloth/JPEGImages/%s.jpg\n" % ( image_id)
        )
        convert_annotation(year, image_id)
    list_file.close()

# os.system("cat 2007_train.txt 2007_val.txt 2012_train.txt 2012_val.txt > train.txt")
# os.system("cat 2007_train.txt 2007_val.txt 2007_test.txt 2012_train.txt 2012_val.txt > train.all.txt")
