import numpy as np
import os
from glob import glob
from lib.dataset.coco_data import BoxInfo
from data_utils import xml_to_coco_json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('--mscocodir', type=str,default='../pubdata/mscoco', help='detect with coco or face',required=False)
# args = parser.parse_args()

# coco_dir=args.mscocodir
folder = '/home/rivian/Desktop/Datasets/derpet_v4_label_tf/'

train_images = glob(os.path.join(folder,"train_images","*.jpg")) + glob(os.path.join(folder,"train_images","*.png"))
val_images = glob(os.path.join(folder,"val_images","*.jpg")) + glob(os.path.join(folder,"val_images","*.png"))

train_annotatons = glob(os.path.join(folder,"train_images","*.xml"))
val_annotations =  glob(os.path.join(folder,"val_images","*.xml"))

train_images = train_images
val_images = val_images

train_outputs = xml_to_coco_json(os.path.join(folder,"train_images"), 'train_output_coco.json')
val_outputs = xml_to_coco_json(os.path.join(folder,"val_images"), 'val_output_coco.json')


train_im_path = os.path.join(folder,'train_images')
train_ann_path =  os.path.join('./','train_output_coco.json')
val_im_path =  os.path.join(folder,'train_images')
val_ann_path =  os.path.join('./','val_output_coco.json')



train_data=BoxInfo(train_im_path,train_ann_path)


fw = open('train.txt', 'w')
for meta in train_data.metas:
    fname, boxes = meta.img_url, meta.bbox



    tmp_str = ''
    tmp_str =tmp_str+ fname+'|'

    for box in boxes:
        data = ' %d,%d,%d,%d,%d'%(box[0], box[1], box[2],  box[3],box[4])
        tmp_str=tmp_str+data
    if len(boxes) == 0:
        print(tmp_str)
        continue
    ####err box?
    if box[2] <= 0 or box[3] <= 0:
        pass
    else:
        fw.write(tmp_str + '\n')
fw.close()






val_data=BoxInfo(val_im_path,val_ann_path)

fw = open('val.txt', 'w')
for meta in val_data.metas:
    fname, boxes = meta.img_url, meta.bbox

    tmp_str = ''
    tmp_str = tmp_str + fname + '|'

    for box in boxes:
        data = ' %d,%d,%d,%d,%d' % (box[0], box[1], box[2], box[3], box[4])
        tmp_str = tmp_str + data
    if len(boxes) == 0:
        print(tmp_str)
        continue
    ####err box?
    if box[2] <= 0 or box[3] <= 0:
        pass
    else:
        fw.write(tmp_str + '\n')
fw.close()
