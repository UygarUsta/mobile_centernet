import sys
sys.path.append('.')
from lib.core.model.centernet_openvino import CenterNet
import numpy as np
import math
import cv2
import torch
import torch.nn.functional as F
import torch.nn as nn
from train_config import config as cfg
import openvino as ov

class Detector:
    def __init__(self,model_path):

        self.device = 'cpu' #torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        self.coreml_ = False
        self.model=CenterNet(inference=False)
        print(model_path)
        state_dict = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        #self.model.cpu()#to(self.device)
        print(self.model.device)
        core = ov.Core()
        input_height = 512
        input_width = 512
        dummy_input = torch.randn(1, 3, cfg.DATA.win, cfg.DATA.hin).to('cpu')
        self.model =  ov.compile_model(ov.convert_model(self.model, example_input=dummy_input))

    def __call__(self, image, score_threshold=0.3,input_shape=(cfg.DATA.hin,cfg.DATA.win),max_boxes=1000):
        """Detect faces.
        Arguments:
            image: a numpy uint8 array with shape [height, width, 3],
                that represents a RGB image.
            score_threshold: a float number.
        Returns:
            boxes: a float numpy array of shape [num_faces, 5].
        """


        if input_shape is None:
            h, w, c = image.shape
            input_shape = (math.ceil(h / 32) * 32, math.ceil(w / 32) * 32)

        else:
            h, w = input_shape
            input_shape = (math.ceil(h / 32) * 32, math.ceil(w / 32) * 32)

        image, scale_x, scale_y, dx, dy = self.preprocess(image,
                                                                 target_height=input_shape[0],
                                                                 target_width=input_shape[1])


        if cfg.DATA.channel==1:
            image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
            image= np.expand_dims(image, -1)

        image_fornet = np.expand_dims(image, 0)
        image_fornet = np.transpose(image_fornet,axes=[0,3,1,2])

        image_fornet=torch.from_numpy(image_fornet).float().to(self.device)
        #with torch.no_grad():
        output =self.model(image_fornet) #[0]
        output_hm = output[0]
        output_wh = output[1]
        #output = torch.tensor(output)
        output_hm = torch.tensor(output_hm)
        output_wh = torch.tensor(output_wh)
        
        output = self.decode(output_hm, output_wh, 4)
        outputs=output.detach().cpu().numpy()
        bboxes=outputs[0]

        bboxes = self.py_nms(np.array(bboxes), iou_thres=None, score_thres=score_threshold,max_boxes=max_boxes)
        
        ###recorver to raw image
        boxes_scaler = np.array([1 / scale_x,
                                 1  / scale_y,
                                 1 / scale_x,
                                 1  / scale_y,
                                 1.,1.], dtype='float32')

        boxes_bias = np.array([dx ,
                               dy ,
                               dx ,
                               dy , 0.,0.], dtype='float32')
                               
        bboxes = (bboxes - boxes_bias)*boxes_scaler


        # self.stats_graph(self._sess.graph)
        return bboxes


    def preprocess(self, image, target_height, target_width, label=None):

        ###sometimes use in objs detects
        h, w, c = image.shape

        bimage = np.zeros(shape=[target_height, target_width, c], dtype=image.dtype)

        scale_y = target_height / h
        scale_x = target_width / w

        scale = min(scale_x, scale_y)

        image = cv2.resize(image, None, fx=scale, fy=scale)

        h_, w_, _ = image.shape

        dx = (target_width - w_) // 2
        dy = (target_height - h_) // 2
        bimage[dy:h_ + dy, dx:w_ + dx, :] = image

        return bimage, scale, scale, dx, dy

    def py_nms(self, bboxes, iou_thres, score_thres, max_boxes=1000):
        upper_thres = np.where(bboxes[:, 4] > score_thres)[0]
        bboxes = bboxes[upper_thres]
        if iou_thres is None:
            return bboxes

        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]

        order = np.argsort(bboxes[:, 4])[::-1]

        keep=[]
        while order.shape[0] > 0:
            if len(keep)>max_boxes:
                break
            cur = order[0]

            keep.append(cur)

            area = (bboxes[cur, 2] - bboxes[cur, 0]) * (bboxes[cur, 3] - bboxes[cur, 1])

            x1_reain = x1[order[1:]]
            y1_reain = y1[order[1:]]
            x2_reain = x2[order[1:]]
            y2_reain = y2[order[1:]]

            xx1 = np.maximum(bboxes[cur, 0], x1_reain)
            yy1 = np.maximum(bboxes[cur, 1], y1_reain)
            xx2 = np.minimum(bboxes[cur, 2], x2_reain)
            yy2 = np.minimum(bboxes[cur, 3], y2_reain)

            intersection = np.maximum(0, yy2 - yy1) * np.maximum(0, xx2 - xx1)

            iou = intersection / (area + (y2_reain - y1_reain) * (x2_reain - x1_reain) - intersection)

            ##keep the low iou
            low_iou_position = np.where(iou < iou_thres)[0]

            order = order[low_iou_position + 1]

        return bboxes[keep]
        
    def decode(self, heatmap, wh, stride, K=100):
        def nms(heat, kernel=3):
            ##fast

            heat = heat.permute([0, 2, 3, 1])
            heat, clses = torch.max(heat, dim=3)

            heat = heat.unsqueeze(1)
            scores = torch.sigmoid(heat)

            hmax = nn.MaxPool2d(kernel, 1, padding=1)(scores)
            keep = (scores == hmax).float()

            return scores * keep, clses
        def get_bboxes(wh):

            ### decode the box
            shifts_x = torch.arange(0, (W - 1) * stride + 1, stride,
                                   dtype=torch.int32)

            shifts_y = torch.arange(0, (H - 1) * stride + 1, stride,
                                   dtype=torch.int32)

            y_range, x_range = torch.meshgrid(shifts_y, shifts_x)

            base_loc = torch.stack((x_range, y_range, x_range, y_range), axis=0)  # (h, w，4)

            base_loc = torch.unsqueeze(base_loc, dim=0).to(self.device)

            wh = wh * torch.tensor([1, 1, -1, -1],requires_grad=False).reshape([1, 4, 1, 1]).to(self.device)
            pred_boxes = base_loc - wh

            return pred_boxes

        batch, cat, H, W = heatmap.size()


        score_map, label_map = nms(heatmap)
        pred_boxes=get_bboxes(wh)


        score_map = torch.reshape(score_map, shape=[batch, -1])

        top_score,top_index=torch.topk(score_map,k=K)

        top_score = torch.unsqueeze(top_score, 2)


        if self.coreml_:
            pred_boxes = torch.reshape(pred_boxes, shape=[batch, 4, -1])
            pred_boxes = pred_boxes.permute([0, 2, 1])
            top_index_bboxes=torch.stack([top_index,top_index,top_index,top_index],dim=2)


            pred_boxes = torch.gather(pred_boxes,dim=1,index=top_index_bboxes)

            label_map = torch.reshape(label_map, shape=[batch, -1])
            label_map = torch.gather(label_map,dim=1,index=top_index)
            label_map = torch.unsqueeze(label_map, 2)

            pred_boxes = pred_boxes.float()
            label_map = label_map.float()


            detections = torch.cat([pred_boxes, top_score, label_map], dim=2)

        else:
            pred_boxes = torch.reshape(pred_boxes, shape=[batch, 4, -1])
            pred_boxes=pred_boxes.permute([0,2,1])

            pred_boxes = pred_boxes[:,top_index[0],:]

            label_map = torch.reshape(label_map, shape=[batch, -1])
            label_map = label_map[:,top_index[0]]
            label_map = torch.unsqueeze(label_map, 2)


            pred_boxes = pred_boxes.float()
            label_map = label_map.float()

            detections = torch.cat([ pred_boxes,top_score, label_map], dim=2)

        return detections

#model = Detector("/home/rivian/Desktop/mobile_centernet/centernet_mobilenetv2_stride4.pth")
model = Detector("./mobile_centernet_fe.pth")
from glob import glob 
import os

folder = "/home/rivian/Desktop/Datasets/derpet_v4_label_tf" #"/home/rivian/Desktop/Datasets/coco_mini_train"
folder = os.path.join(folder,"val_images") #place to val

files = glob(folder+"/*.jpg") + glob(folder+"/*.png")
for i in files:
    #image,annos = infer_image(model,i,classes,conf,half,input_shape=(416,416),cpu=cpu,openvino_exp=openvino_exp)
    image = cv2.imread(i)
    bboxes = model(image)
    for i in bboxes:
        xmin = int(i[0])
        ymin = int(i[1])
        xmax = int(i[2])
        ymax = int(i[3])
        name = int(i[-1])
        cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(0,255,0),3)
        cv2.putText(image,str(name),(xmin-3,ymin),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),2)

    cv2.imshow("img",image)
    ch = cv2.waitKey(0)
    if ch == ord("q"): break


video_path = "/home/asis/Desktop/vlc-record-2024-10-08-17h10m15s-2_18.00.00_novis_output.avi-.avi"
#video_path = 0
import time
cap = cv2.VideoCapture(video_path)
while 1:
    ret,image = cap.read()
    fps1 = time.time()
    bboxes = model(image)
    fps2 = time.time()
    for i in bboxes:
        xmin = int(i[0])
        ymin = int(i[1])
        xmax = int(i[2])
        ymax = int(i[3])
        score = float(i[4])
        name = int(i[-1])
        cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(0,255,0),3)
        cv2.putText(image,str(name),(xmin-3,ymin),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),2)
        cv2.putText(image,f'{score:.2f}',(xmax-3,ymin),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),2)
    fps = 1 / (fps2-fps1)
    cv2.putText(image,f'FPS:{fps:.2f}',(200,100),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),2)
    cv2.imshow("img",image)
    ch = cv2.waitKey(1)
    if ch == ord("q"): break
