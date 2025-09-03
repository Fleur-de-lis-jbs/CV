#颜色检测器
import numpy as np
import cv2
from PIL import Image
def get_limits(color):
    # 步骤1：将输入颜色转换为单像素BGR图像
    c = np.uint8([[color]])  # 构造形状为 (1,1,3) 的数组，模拟一个像素的BGR图像
    
    # 步骤2：将BGR颜色转换为HSV颜色空间
    hsv_c = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)  # 转换后得到该颜色的HSV值
    
    # 步骤3：计算H通道的范围（以中心值±10）
    h = hsv_c[0][0][0]  # 提取HSV中H通道的值（色调）
    lower_h = h - 10    # H通道下限
    upper_h = h + 10    # H通道上限
    
    # 步骤4：固定S和V通道的范围
    lower_s, upper_s = 100, 255  # 饱和度范围（排除过灰的颜色）
    lower_v, upper_v = 100, 255  # 明度范围（排除过暗的颜色）
    
    # 步骤5：构造HSV上下限数组
    lower_limit = np.array([lower_h, lower_s, lower_v], dtype=np.uint8)
    upper_limit = np.array([upper_h, upper_s, upper_v], dtype=np.uint8)
    
    return lower_limit, upper_limit  # 返回HSV阈值范围

webcam = cv2.VideoCapture(0)#一个摄像头，编号默认为0
yellow = [0,255,255]
color = yellow

lowerlimit,upperlimit = get_limits(color)
while True:
    ret,frame = webcam.read()
    hsvImage = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)#转化为HSV色彩空间
    mask = cv2.inRange(hsvImage,lowerlimit,upperlimit)
    mask_ = Image.fromarray(mask)#转化为pillow图像
    bbox = mask_.getbbox()#返回边界框，四个坐标
    if bbox is not None:
        x1,y1,x2,y2 = bbox
        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
    cv2.imshow('frame',frame)
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break
webcam.release()
