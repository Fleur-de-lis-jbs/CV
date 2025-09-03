#图像分类器
from img2vec_pytorch import Img2Vec#用于获取特征
import os 
import shutil
from sklearn.model_selection import train_test_split#用于划分训练集和验证集
from PIL import Image
from sklearn.ensemble import RandomForestClassifier#随机森林模型，做分类器
from sklearn.metrics import accuracy_score
import pickle

img2vec = Img2Vec()#特征提取器
data_dir = r'C:\Users\13394\Desktop\项目\CV\day5\weather_recognition-main\weather_recognition-main\data'
img_dir = os.path.join(data_dir, 'img')

train_dir = os.path.join(data_dir,'train')
val_dir = os.path.join(data_dir,'val')

# #创建文件夹
# os.makedirs(train_dir, exist_ok=True)
# os.makedirs(val_dir, exist_ok=True)

# #划分验证集和训练集
# for category in os.listdir(img_dir):
#     category_dir = os.path.join(img_dir, category)
#     img_paths = [os.path.join(category_dir, img) for img in os.listdir(category_dir) if img.endswith(('.jpg', '.jpeg', '.png'))]#获得所有图像的路径
#     # 划分训练集和验证集，这里按 8:2 划分
#     train_paths, val_paths = train_test_split(img_paths, test_size=0.2, random_state=42)    
    
#     train_category_dir = os.path.join(train_dir, category)
#     val_category_dir = os.path.join(val_dir, category)
#     os.makedirs(train_category_dir, exist_ok=True)
#     os.makedirs(val_category_dir, exist_ok=True)

#         # 复制训练集图像到目标目录
#     for img_path in train_paths:
#         shutil.copy(img_path, train_category_dir)
    
#     # 复制验证集图像到目标目录
#     for img_path in val_paths:
#         shutil.copy(img_path, val_category_dir)

# print("训练集和验证集划分完成！")

data = {}
for j,dir_ in enumerate([train_dir,val_dir]):
    features = []
    labels = []

    for category in os.listdir(dir_):
         for image_path in os.listdir(os.path.join(dir_,category)):
            image_path_ = os.path.join(dir_,category,image_path)
            img = Image.open(image_path_)
            img_features = img2vec.get_vec(img)
            
            features.append(img_features)
            labels.append(category)
    data [['training_data','validation_data'][j]] = features
    data [['training_label','validation_label'][j]] = labels

#训练模型
model = RandomForestClassifier()
model.fit(data['training_data'],data['training_label'])

#验证数据
y_pred = model.predict(data['validation_data'])
score = accuracy_score(y_pred , data['validation_label'])#计算准确率分数
print(score)

#保存模型
with open (r'C:\Users\13394\Desktop\项目\CV\day5\model.p','wb') as f:
    pickle.dump(model,f)
