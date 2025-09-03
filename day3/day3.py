#面部匿名化,图片和视频
import cv2
import mediapipe as mp
import numpy as np
import argparse

def process_image(img,face_detection):
    #导入图片
    if img is None or img.size == 0:
         return img
    H,W,_ = img.shape

    img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)#默认BGR，meidiapipe库要用RGB
    out = face_detection.process(img_RGB)#执行人脸检测,返回列表，列表内每个元素的值表示一张人脸，包含边界框和人脸上的六个定位信息
    if out.detections is not None:   #如果没有检测到人脸，out.detection 返回None
        for i in out.detections:
            location_data = i.location_data
            bbox = location_data.relative_bounding_box
            x1,y1,w,h = bbox.xmin,bbox.ymin,bbox.width,bbox.height#人脸边界框,返回的是比例,相对值
            x1 = int(x1*W)
            y1 = int(y1*H)
            w  = int(w*W)
            h  = int(h*H)

            x1 = max(0, x1)
            y1 = max(0, y1)
            w = min(W - x1, w)
            h = min(H - y1, h)
            if w > 0 and h > 0:
                img = cv2.rectangle(img,(x1,y1),(x1+w,y1+h),(255,0,0),10)
                #模糊人脸
                img[y1:y1+h,x1:x1+w,:] = cv2.blur(img[y1:y1+h,x1:x1+w,:],(30,30))#最后一个冒号表示所有通道
    return img
args = argparse.ArgumentParser()#创建解析器对象
args.add_argument('--mode', default='video')
args.add_argument('--filePath',default=r'C:\Users\13394\Desktop\项目\CV\视频 人脸识别.mp4')
args = args.parse_args()
#人脸检测
mp_face_detection = mp.solutions.face_detection#导入人脸检测功能模块
with mp_face_detection.FaceDetection(min_detection_confidence = 0.3,model_selection = 1) as face_detection:#两个参数分别为置信度和检测模型，检测模型的不同与检测距离有关.0是两米内，1是五米外
    if args.mode in ['image']:
        img = cv2.imread(r'C:\Users\13394\Desktop\项目\CV\单人照 人脸识别.jpg')
        img = process_image(img,face_detection = face_detection)
        cv2.imwrite(r'C:\Users\13394\Desktop\项目\CV\output.jpg',img)
    elif args.mode in ['video']:
        
        cap = cv2.VideoCapture(args.filePath)
        ret,frame = cap.read()
        if not ret or frame is None:
                print("错误：无法读取视频帧（视频损坏或格式不支持）")
                cap.release()
        output_video = cv2.VideoWriter(r'C:\Users\13394\Desktop\项目\CV\output.mp4',cv2.VideoWriter_fourcc(*'mp4v'),25,(frame.shape[1],frame.shape[0]))
        while ret:
            processed_frame = process_image(frame,face_detection = face_detection)
            if processed_frame is not None:
                output_video.write(processed_frame)            
            ret,frame = cap.read()

        cap.release()
        output_video.release()