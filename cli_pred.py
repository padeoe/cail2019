from model import BertSimMatchModel

MODEL_DIR = "model"
model = BertSimMatchModel.load(MODEL_DIR)

while True:
    text = input("输入句子: ")
    a, b, c, *_ = text.split()
    results = model.predict([(a, b, c)])

    for label, prob in results:
        print(str(label), prob)
