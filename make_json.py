import glob
import os
import json

file_list = glob.glob('images/*.png')
name_list = [os.path.splitext(os.path.basename(file))[0] for file in file_list]

train = []
test = []

count = 0
for name in name_list:
    if count % 2 == 0:
        train.append(name)
    else:
        test.append(name)
    count += 1

with open("train.json", "w") as f:
    json.dump(train, f, indent=2)

with open("test.json", "w") as f:
    json.dump(test, f, indent=2)