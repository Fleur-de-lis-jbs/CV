import cv2
import os
import numpy as np
# #图像

# #read
# image = cv2.imread(r'C:\Users\13394\Desktop\项目\CV\AnxixiangAnxixiangAnxixiang21.jpg')#图像是一个numpy数组

# #write
# cv2.imwrite('123.jpg',image)

# #visualize
# cv2.imshow('img',image)# img为窗口的名称
# cv2.waitKey(0) #停顿，直到关闭

# #视频

# #read
# video_path = os.path.join('.','data','monkey.mp4')#没有实际文件
# video = cv2.VideoCapture(video_path)

# #visualize
# ret = True #有帧还没有读取，则为True
# while ret:
#     ret,frame = video.read()
#     if ret:
#         cv2.imshow('frame',frame)
#         cv2.waitKey(40)#每一帧之后40毫秒，根据实际帧数决定

# video.release()
# cv2.destroyAllWindows()#释放内存

# #连接摄像头,read
# webcam = cv2.VideoCapture(0)#一个摄像头，编号默认为0

# #visualize
# while True:
#     ret,frame = webcam.read()
#     cv2.imshow('frame',frame)
#     if cv2.waitKey(40) & 0xFF == ord('q'):
#         break

# webcam.release()
# cv2.destroyAllWindows()#释放内存

# #调整图像大小
# image = cv2.imread(r'C:\Users\13394\Desktop\项目\CV\AnxixiangAnxixiangAnxixiang21.jpg')
# resized_image = cv2.resize(image,(1000,1000))#调整至新的大小,先高后宽
# print(image.shape)
# print(resized_image.shape)
# cv2.imshow('image',image)
# cv2.imshow('reimage',resized_image)
# cv2.waitKey(0)

# #crop,裁剪图像
# image = cv2.imread(r'C:\Users\13394\Desktop\项目\CV\AnxixiangAnxixiangAnxixiang21.jpg')
# print(image.shape)
# cropped_image = image[100:400,100:400]#先高后宽
# cv2.imshow('image',image)
# cv2.imshow('cimage',cropped_image)
# cv2.waitKey(0)

# #色彩空间
# image = cv2.imread(r'C:\Users\13394\Desktop\项目\CV\AnxixiangAnxixiangAnxixiang21.jpg')#图像是一个numpy数组

# #BGR变为RGB
# img_RGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
# #变为灰度图
# img_grav = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# #变为hsv
# img_hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
# cv2.imshow('img',image)# img为窗口的名称
# cv2.imshow('img1',img_RGB)
# cv2.imshow('img2',img_grav)
# cv2.imshow('img3',img_hsv)
# cv2.waitKey(0) #停顿，直到关闭

# #模糊处理,可以用来去除噪声
# image = cv2.imread(r'C:\Users\13394\Desktop\项目\CV\AnxixiangAnxixiangAnxixiang21.jpg')#图像是一个numpy数组
# ksize = 9 #模糊核的大小,越大模糊效果越强，以元组传入

# image_blur = cv2.blur(image,(ksize,ksize))#经典模糊(均值模糊)，模糊核无要求

# image_gaussian_blur = cv2.GaussianBlur(image,(ksize,ksize),3)# 3为x方向高斯标准差，y方向不指定默认与x方向相同，模糊核得是奇数

# image_medain_blur = cv2.medianBlur(image,ksize)#中值模糊,模糊核不以元组传入

# cv2.imshow('img1',image)
# cv2.imshow('img2',image_blur)
# cv2.imshow('img3',image_gaussian_blur)
# cv2.imshow('img4',image_medain_blur)
# cv2.waitKey(0)

# #图像阈值处理,单一阈值
# image = cv2.imread(r'C:\Users\13394\Desktop\项目\CV\AnxixiangAnxixiangAnxixiang21.jpg')
# image_gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)

# ret,dst = cv2.threshold(image_gray,80,255,cv2.THRESH_BINARY)#处理灰度图，80为阈值，高于80的设置为255，低于80为0,THRESH_BINARY为阈值化类型
# dst1 = cv2.blur(dst,(10,10))
# ret,dst2 = cv2.threshold(dst1,80,255,cv2.THRESH_BINARY)#处理灰度图，80为阈值，高于80的设置为255，低于80为0,THRESH_BINARY为阈值化类型


# cv2.imshow('img1',image)
# cv2.imshow('img2',dst)
# cv2.imshow('img3',dst1)

# cv2.waitKey(0)

# #图像阈值处理,自适应阈值,可以用于去除图像阴影等
# image = cv2.imread(r'C:\Users\13394\Desktop\项目\CV\AnxixiangAnxixiangAnxixiang21.jpg')
# image_gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)

# #adaptiveMethod：自适应阈值的计算方法（两种可选）：
# #cv2.ADAPTIVE_THRESH_MEAN_C：以邻域内像素的平均值作为阈值（减去常数 C）
# #cv2.ADAPTIVE_THRESH_GAUSSIAN_C：以邻域内像素的高斯加权平均值作为阈值（减去常数 C）

# dst = cv2.adaptiveThreshold(image_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,5,30 )#cv2.THRESH_BINARY 是阈值化类型,5是阈值的邻域大小，得为奇数，30是常数


# cv2.imshow('img1',image)
# cv2.imshow('img2',dst)

# cv2.waitKey(0)

# #边缘检测
# image = cv2.imread(r'C:\Users\13394\Desktop\项目\CV\AnxixiangAnxixiangAnxixiang21.jpg')
# image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# edge = cv2.Canny(image,100,200)#传入灰度图，普通图片都可以

# kernel = np.ones((5, 5), np.uint8)
# edge_d = cv2.dilate(edge ,kernel)  #图像膨胀函数,边缘变粗

# edge_e = cv2.erode(edge ,kernel)#腐蚀边缘，与膨胀相反，边缘变细
# cv2.imshow('img1',image)
# cv2.imshow('img2',edge)     
# cv2.imshow('img3',edge_d)       
# cv2.imshow('img4',edge_e)
# cv2.waitKey(0)

# #图像绘制
# image = cv2.imread(r'C:\Users\13394\Desktop\项目\CV\AnxixiangAnxixiangAnxixiang21.jpg')

# #line
# cv2.line(image,(200,200),(100,100),(0,255,0))
# #rectangle
# cv2.rectangle(image,(200,200),(100,100),(0,0,255))
# #circle
# cv2.circle(image,(150,150),50,(255,0,0),-1)#当图形的线的厚度是-1时，图形会被填充
# #text
# cv2.putText(image,'Hello',(300,300),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))

# cv2.imshow('img1',image)
# cv2.waitKey(0)

#轮廓处理
image = cv2.imread(r'C:\Users\13394\Desktop\项目\CV\AnxixiangAnxixiangAnxixiang21.jpg')
img_grav = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
ret,dst = cv2.threshold(img_grav,127,255,cv2.THRESH_BINARY_INV)#反向二值化处理
contours, hierarchy = cv2.findContours(dst,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)#第二三个参数时轮廓检索模式和轮廓近似方法,contours是numpy数组，表示所有轮廓
#简单的物体检测器
for cnt in contours:
    if (cv2.contourArea(cnt))>200:#轮廓的面积值#排除噪声
        cv2.drawContours(image,cnt,-1,(255,0,0),1)
        x1,y1,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(image,(x1,y1),(x1+w,y1+h),(0,255,0),2)
cv2.imshow('img1',image)
cv2.imshow('img2',dst)
cv2.waitKey(0)
