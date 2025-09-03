#测试模型
from img2vec_pytorch import Img2Vec
from PIL import Image
import pickle

with open (r'C:\Users\13394\Desktop\项目\CV\day5\model.p','rb') as f:
    model = pickle.load(f)
img2vec = Img2Vec()

image_path  = r'C:\Users\13394\Desktop\项目\CV\day5\weather_recognition-main\weather_recognition-main\data\img\colddamage\02.jpg'

img = Image.open(image_path)

features = img2vec.get_vec(img)

pred = model.predict([features])

print(pred)