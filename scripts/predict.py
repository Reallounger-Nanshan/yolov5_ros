#!/home/hxl228server20/anaconda3/envs/yolov5/bin/python3.9

import rospy
import os
import platform
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import time

from models.common import DetectMultiBackend
from utils.general import (LOGGER, Profile, increment_path, non_max_suppression, scale_boxes, xyxy2xywh)
from utils.torch_utils import smart_inference_mode

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from yolov5_ros.msg import Result


class Predictor():
    def __init__(self):
        # Get python file path
        file_path = os.path.dirname(__file__)
        project_path = file_path[:file_path.rfind("/") + 1]

        # Set parameters
        self.weights = project_path + "weights/yolov5s.pt"  # model path or triton URL
        self.data = project_path + "cfg/datasets/coco128.yaml"  # dataset.yaml path
        self.imgsz = 640  # inference size (height, width)
        self.conf_thres = 0.25  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 1000  # maximum detections per image
        self.classes = None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False  # class-agnostic NMS
        self.augment = False  # augmented inference
        self.half = False  # use FP16 half-precision inference
        self.dnn = False  # use OpenCV DNN for ONNX inference
        self.color_padding = (114, 114, 114)
        self.is_xywh = True
        self.show_img = False
        self.color_plot = [(227, 23, 13), (255, 255, 0), (255, 0, 255), (64, 224, 205), (0, 0, 255),
                      (34, 139, 34), (3, 168, 158), (138, 43, 226), (237, 145, 33), (128, 42, 42)
        ]  # RGB
        self.thickness = 2
        self.batch_size = 1
        self.stride = 32

        self.bridge = CvBridge()

        # Create model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = DetectMultiBackend(
            self.weights,
            device = self.device,
            dnn = self.dnn,
            data = self.data,
            fp16 = self.half
            )

        # Warmup
        self.model.warmup(imgsz = (1 if self.model.pt or self.model.triton else self.batch_size, 3, self.imgsz, self.imgsz))


        # ROS publisher
        self.result_pub = rospy.Publisher("/yolov5_predict_node/result", Result, queue_size = 10)


    def LetterBox(self, img_lb):
        # Scale ratio (new / old)
        # image shape: height, width, channel
        ratio = min(self.imgsz / img_lb.shape[0], self.imgsz / img_lb.shape[1])

        # Compute padding
        dw, dh = self.imgsz - img_lb.shape[1], self.imgsz - img_lb.shape[0]  # wh padding
        if self.model.pt:  # Minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding

        # Divide padding into 2 sides
        dw /= 2
        dh /= 2

        # Resize image
        if ratio == 1.0:
            new_shape = int(round(img_lb.shape[0] * ratio)), int(round(img_lb.shape[1] * ratio))
            img_lb = cv2.resize(img_lb, new_shape, interpolation = cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        # Add border
        img_lb = cv2.copyMakeBorder(img_lb, top, bottom, left, right, cv2.BORDER_CONSTANT, value = self.color_padding)
        
        return img_lb

    
    def LoadImage(self, ros_img_msg):
        # ROS Image to OpenCV Image
        cv_img = self.bridge.imgmsg_to_cv2(ros_img_msg, desired_encoding = "bgr8")

        img_padding = self.LetterBox(cv_img)  # Padded resize
        img_padding = img_padding.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img_padding = np.ascontiguousarray(img_padding)  # Contiguous

        return img_padding, cv_img


    def PreProcess(self, img):
        img_pre = torch.from_numpy(img).to(self.model.device)
        img_pre = img_pre.half() if self.model.fp16 else img_pre.float()  # uint8 to fp16/32
        img_pre /= 255  # 0 - 255 to 0.0 - 1.0
        if len(img_pre.shape) == 3:
            img_pre = img_pre[None]  # Expand for batch dim

        return img_pre


    def PostProcess(self, im, det, img_raw_shape):
        # NMS
        det = non_max_suppression(
            det, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det = self.max_det
            )[0]

        # Calculate the position of the bbox on the raw image
        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img_raw_shape).round()

        return det


    def PublishResult(self, predict_results, image = None):
        result = Result()
        if self.is_xywh:
            cls_num, cls_name, confidence, c_x, c_y, w, h = [], [], [], [], [], [], []
        else:
            cls_num, cls_name, confidence, tl_x, tl_y, br_x, br_y = [], [], [], [], [], [], []

        for *xyxy, conf, cls in reversed(predict_results):
            cls_num.append(int(cls))
            cls_name.append(self.model.names[int(cls)])
            confidence.append(float(conf))

            if self.is_xywh:
                xywh = xyxy2xywh(torch.tensor(xyxy).view(1, 4))
                c_x.append(float(xywh[0][0]))
                c_y.append(float(xywh[0][1]))
                w.append(float(xywh[0][2]))
                h.append(float(xywh[0][3]))
            else:
                tl_x.append(float(xyxy[0]))
                tl_y.append(float(xyxy[1]))
                br_x.append(float(xyxy[2]))
                br_y.append(float(xyxy[3]))

            if self.show_img:
                if self.is_xywh:
                    top_left = (int(xywh[0][0] - xywh[0][2] / 2), int(xywh[0][1] - xywh[0][3] / 2))
                    bottom_right = (int(xywh[0][0] + xywh[0][2] / 2), int(xywh[0][1] + xywh[0][3] / 2))
                else:
                    top_left = (int(xyxy[0]), int(xyxy[1]))
                    bottom_right = (int(xyxy[2]), int(xyxy[3]))

                color = self.color_plot[int(cls) % 10]

                cv2.rectangle(image, top_left, bottom_right, color, self.thickness)
                        
                text = str(int(cls)) + " " + self.model.names[int(cls)]
                top_left_txt = (int(top_left[0]), int(top_left[1] - 20))
                cv2.putText(image, text, top_left_txt, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 4)
 
        result.classes_num, result.classes_name , result.confidence = cls_num, cls_name, confidence
        if self.is_xywh:
            result.centre_x , result.centre_y , result.width , result.height = c_x, c_y, w, h
        else:
            result.top_left_x , result.top_left_y = tl_x, tl_y
            result.bottom_right_x , result.bottom_right_y = br_x, br_y
        
        self.result_pub.publish(result)

        return image
        

    @smart_inference_mode()
    def Detect(self, ros_img_msg):
        time_start = time.time()

        # Get image
        cudnn.benchmark = True
        img, img_raw = self.LoadImage(ros_img_msg)

        dt = (Profile(), Profile(), Profile())

        # Preprocess
        with dt[0]:
            img_pre = self.PreProcess(img)

        # Inference
        with dt[1]:
            # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = self.model(img_pre, augment = self.augment)

        # Postprocess
        with dt[2]:
            predict_results = self.PostProcess(img_pre, pred, img_raw.shape)

        if self.show_img:
            img_plt = img_raw.copy()

        # Publish ROS topic
        if predict_results.shape[0]:
            if self.show_img:
                img_plt = self.PublishResult(predict_results, image = img_plt)
            else:
                self.PublishResult(predict_results)

        if self.show_img:
            cv2.imshow("detect", img_plt)
            cv2.waitKey(1)

        # Calculate frame rate
        time_end = time.time()
        fps = 1 / (time_end - time_start)
        print("\n", "%.2f"%fps, "FPS\n")

        t = tuple(x.t / 1 * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, self.imgsz, self.imgsz)}' % t)


if __name__=="__main__":
    predictor = Predictor()

    # Create ROS node
    rospy.init_node("yolov5_predict_node")
    # ROS subscriber
    rospy.Subscriber("/webcam_imgmsg", Image, predictor.Detect)

    rospy.spin()
