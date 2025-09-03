import cv2
import mediapipe as mp
import os 
import numpy as np

def get_face_landmarks (img,face_mesh,draw = False,static_image_mode = True):
    #读入图片
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    img_rows , img_cols , _ =img.shape
    results = face_mesh.process(img_rgb)#返回包含人脸信息关键点的对象

    img_landmarks = []

    if results.multi_face_landmarks:#检测到人脸时继续
        if draw:#如果要绘制人脸
            mp_drawing = mp.solutions.drawing_utils 
            mp_drawing.styles = mp.solutions.drawing_styles
            drawing_spec = mp_drawing.DrawingSpec(thickness = 1 , circle_radius = 0.5)

            mp_drawing.draw_landmarks(
                image = img,
                landmark_list = results.multi_face_landmarks[0],
                connections = mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec = drawing_spec,
                connection_drawing_spec = drawing_spec
            )
        ls_single_face = results.multi_face_landmarks[0].landmark
        xs_ = []
        ys_ = []
        zs_ = []
        for idx in ls_single_face:
            xs_.append(idx.x)
            ys_.append(idx.y)
            zs_.append(idx.z)
        for j in range(len(xs_)):
            img_landmarks.append(xs_[j] - min(xs_))
            img_landmarks.append(ys_[j] - min(ys_))
            img_landmarks.append(zs_[j] - min(zs_))#归一化

    return img_landmarks

data_dir = r'C:\Users\13394\Desktop\项目\CV\day6\RAF-DB人脸表情数据集\RAF-DB人脸表情数据集\PublicTest'
output = []
face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode = True , max_num_faces = 1 ,min_detection_confidence = 0.5)#初始化facemesh 模型 ，static_image_mode = False 表示处理视频流
for emotion_index,emotion in enumerate(os.listdir(data_dir)):
    for image_path_ in os.listdir(os.path.join(data_dir,emotion)):
        image_path = os.path.join(data_dir,emotion,image_path_)
        
        image = cv2.imread(image_path)
        face_landmarks = get_face_landmarks(image,face_mesh)

        if len(face_landmarks) == 1404:
            face_landmarks.append(int(emotion_index))
            output.append(face_landmarks)

np.savetxt('data.txt',np.asarray(output))
face_mesh.close()