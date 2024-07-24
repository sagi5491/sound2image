import json
from PIL import Image

with open("MUSIC21_solo_videos.json") as f:
    di = json.load(f)

train = {}
test = {}

for e in di["videos"]:
    count = 0
    train[e] = []
    test[e] = []
    if(e == "guzheng"): break
    for f in di["videos"][e]:
        for i in range(20):
            try:
                Image.open("melspectrograms/" + e + "/" + f + "/" + str(i) + ".png")
            except FileNotFoundError as err:
                print(err)
            else:
                if(count % 9 != 0):
                    train[e].append(f + "/" + str(i))
                else:
                    test[e].append(f + "/" + str(i))
        count += 1

with open("train2.json", "w") as f:
    json.dump(train, f, indent=2)

with open("test2.json", "w") as f:
    json.dump(test, f, indent=2)