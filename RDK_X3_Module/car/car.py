#!/usr/bin/env python3

import numpy as np
import cv2
from postprocess import postprocess

from hobot_dnn import pyeasy_dnn as dnn
from hobot_vio import libsrcampy as srcampy
import Hobot.GPIO as GPIO
import time 

pwma , pwmb = 32 ,33

stby , ain2 , ain1 , bin1 , bin2 = 11 , 13 , 15 , 16 , 18

GPIO.setmode(GPIO.BOARD)

GPIO.setup(stby,GPIO.OUT)
GPIO.setup(ain1,GPIO.OUT)
GPIO.setup(ain2,GPIO.OUT)
GPIO.setup(bin1,GPIO.OUT)
GPIO.setup(bin2,GPIO.OUT)

GPIO.setmode(GPIO.BOARD)

pa = GPIO.PWM(pwma,48000)
pb = GPIO.PWM(pwmb,48000)

GPIO.output(stby,GPIO.HIGH)

pa.start(100)
pb.start(100)
pa.ChangeDutyCycle(100)
pb.ChangeDutyCycle(100)
pa.start(100)
pb.start(100)
    
def ctrl(left,right):
    if left >= 0:
        GPIO.output(ain1,GPIO.HIGH)
        GPIO.output(ain2,GPIO.LOW)
    else:
        GPIO.output(ain2,GPIO.HIGH)
        GPIO.output(ain1,GPIO.LOW)
        
    if right >= 0:
        GPIO.output(bin1,GPIO.HIGH)
        GPIO.output(bin2,GPIO.LOW)
    else:
        GPIO.output(bin2,GPIO.HIGH)
        GPIO.output(bin1,GPIO.LOW)
        
    pa.ChangeDutyCycle(abs(left))
    pb.ChangeDutyCycle(abs(right))
    pa.start(abs(left))
    pb.start(abs(right))
    
def stop():
    pa.stop()
    pb.stop()
    GPIO.output(stby,GPIO.LOW)
    GPIO.output(ain1,GPIO.LOW)
    GPIO.output(ain2,GPIO.LOW)
    GPIO.output(bin1,GPIO.LOW)
    GPIO.output(bin2,GPIO.LOW)

def bgr2nv12_opencv(image):
    height, width = image.shape[0], image.shape[1]
    area = height * width
    yuv420p = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))
    y = yuv420p[:area]
    uv_planar = yuv420p[area:].reshape((2, area // 4))
    uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))

    nv12 = np.zeros_like(yuv420p)
    nv12[:height * width] = y
    nv12[height * width:] = uv_packed
    return nv12


def get_hw(pro):
    if pro.layout == "NCHW":
        return pro.shape[2], pro.shape[3]
    else:
        return pro.shape[1], pro.shape[2]


def print_properties(pro):
    print("tensor type:", pro.tensor_type)
    print("data type:", pro.dtype)
    print("layout:", pro.layout)
    print("shape:", pro.shape)
    
    
class PID():
    def __init__(self,KP,KI,KD):
        self.KP = KP
        self.KI = KI
        self.KD = KD
        self.miss = [0,0,0]
        self.time = [time.time()-1, time.time()-0.5, time.time()]
        self.miss_i = 0
        
    def miss_in(self,miss):
        del self.miss[0]
        self.miss.append(miss)
        del self.time[0]
        self.time.append(time.time())
        
        self.miss_i += (self.miss[2] - self.miss[1])*(self.time[2] - self.time[1])
        
        #if self.miss_i > 1:
        #    self.miss_i = 1
        #if self.miss_i < -1:
        #    self.miss_i = -1
        
    def m_out(self):
        return self.KP * self.miss[2] + self.KI * self.miss_i + self.KD * (self.miss[2] - self.miss[1]) / (self.time[2] - self.time[1])
        
    def zero(self):
        self.miss = [0,0,0]
        self.time = [time.time()-1, time.time()-0.5, time.time()]
        self.miss_i = 0

if __name__ == '__main__':
    models = dnn.load('/root/telecar/hand_yolov5.bin')
    # 打印输入 tensor 的属性
    print_properties(models[0].inputs[0].properties)
    # 打印输出 tensor 的属性
    print(len(models[0].outputs))
    for output in models[0].outputs:
        print_properties(output.properties)

    #cam = srcampy.Camera()
    #cam.open_cam(0, -1, 30, 640, 640)
    
    video = cv2.VideoCapture(8)
    print(video.isOpened())
    codec = cv2.VideoWriter_fourcc( 'M', 'J', 'P', 'G' )
    video.set(cv2.CAP_PROP_FOURCC, codec)
    video.set(cv2.CAP_PROP_FPS, 30)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    pid = PID(10,0.65,0.005)
    
    try:
        while True:
            # img_file = cv2.imread('./hand_test4.jpg')
            h, w = get_hw(models[0].inputs[0].properties)
            des_dim = (w, h)
            
            _ , img_file = video.read()
            
            #img = cam.get_img(2, 640, 640)
            #img = np.frombuffer(img, dtype=np.uint8)
            resized_data = cv2.resize(img_file, des_dim, interpolation=cv2.INTER_AREA)
            nv12_data = bgr2nv12_opencv(resized_data)

            outputs = models[0].forward(nv12_data)
            
            print('-'*50)
            prediction_bbox = postprocess(outputs, model_hw_shape=(640, 640), origin_img_shape=(20,20))
            print(prediction_bbox)
            
            max_data = 0
            max_i = -1
            for i in range(len(prediction_bbox)):
                if prediction_bbox[i][4] > max_data:
                    max_data = prediction_bbox[i][4]
                    max_i = i
                    
            ctrl(0,0)
                    
            if max_i>=0:
                miss = (prediction_bbox[max_i][0]+prediction_bbox[max_i][2])/2 - 10
                pid.miss_in(miss)
                pwm_m = int(pid.m_out())
                pwml = 0 - pwm_m
                pwmr = 0 + pwm_m
                if pwml > 40:
                    pwml = 40
                if pwmr > 40:
                    pwmr = 40
                if pwml < -40:
                    pwml = -40
                if pwmr < -40:
                    pwmr = -40
                    
                ctrl(pwml,pwmr)
                
            else:
                pid.zero()
                pwm_m = int(pid.m_out())
                pwml = 0 - pwm_m
                pwmr = 0 + pwm_m
                if pwml > 40:
                    pwml = 40
                if pwmr > 40:
                    pwmr = 40
                if pwml < -40:
                    pwml = -40
                if pwmr < -40:
                    pwmr = -40
                    
                ctrl(pwml,pwmr)

                
            
                
                
            
    finally:
        #cam.close_cam()
        stop()
        video.release()
