#文本检测
import pytesseract
from PIL import Image
from easyocr import Reader

pytesseract.pytesseract.tesseract_cmd = r"D:\Tessseract\tesseract.exe"
image_path = r"C:\Users\13394\Desktop\项目\CV\文本检测数据集\you_are_beautiful.jpg"
# text = pytesseract.image_to_string(Image.open(image_path),lang='eng')
# print(text)#逻辑没问题，文件和环境配置变量也没问题，但不知道为什么输出没有结果

# text =''
# reader = Reader(['en'])
# results = reader.readtext(Image.open(image_path))#使用easyocr库的时候开代理，要在网上下载模型
# for result in results:
#     text = text +result[1] +' '
# text = text[:-1]
# print(text)

def jaccard_similarity(sentence1, sentence2):#计算杰卡德相似度，用于测量比较文本检测结果
    # Tokenize sentences into sets of words
    set1 = set(sentence1.lower().split())
    set2 = set(sentence2.lower().split())

    # Calculate Jaccard similarity
    intersection_size = len(set1.intersection(set2))
    union_size = len(set1.union(set2))

    # Avoid division by zero if both sets are empty
    similarity = intersection_size / union_size if union_size != 0 else 0.0

    return similarity