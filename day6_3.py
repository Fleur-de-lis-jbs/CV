#测试模型
import cv2
import pickle
from day6_1 import get_face_landmarks
import mediapipe as mp

with open(r'C:\Users\13394\Desktop\项目\CV\day6\model','rb') as f:
    model = pickle.load(f)
cap = cv2.VideoCapture(0)

ret , frame = cap.read()
face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode = False , max_num_faces = 1 ,min_detection_confidence = 0.5 , refine_landmarks=False)#初始化facemesh 模型 ，static_image_mode = False 表示处理视频流
while ret:
    ret,frame = cap.read()
    if not ret:
        break
    face_landmarks = get_face_landmarks(frame,face_mesh=face_mesh )
    if not face_landmarks:
        print("⚠️ 未提取到关键点（可能无人脸）")
    else:
        print(f"ℹ️ 提取到关键点，长度：{len(face_landmarks)}")  # 正常应为2067
        output= model.predict([face_landmarks])
        print(output)

    cv2.imshow('frame',frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
cap.release()
