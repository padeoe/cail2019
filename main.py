import json
import logging
import sys
import time

import torch

from model import BertSimMatchModel

logging.disable(sys.maxsize)

start_time = time.time()
input_path = "/input/input.txt"
output_path = "/output/output.txt"

if len(sys.argv) == 3:
    input_path = sys.argv[1]
    output_path = sys.argv[2]

inf = open(input_path, "r", encoding="utf-8")
ouf = open(output_path, "w", encoding="utf-8")

MODEL_DIR = "model"
model = BertSimMatchModel.load(MODEL_DIR, torch.device("cpu"))

text_tuple_list = []
for line in inf:
    line = line.strip()
    items = json.loads(line)
    a = items["A"]
    b = items["B"]
    c = items["C"]
    text_tuple_list.append((a, b, c))

results = model.predict(text_tuple_list)

for label, _ in results:
    # print(str(label), _)
    print(str(label), file=ouf)

inf.close()
ouf.close()

end_time = time.time()
spent = end_time - start_time
print("numbers of samples: %d" % len(results))
print("time spent: %.2f seconds" % spent)
