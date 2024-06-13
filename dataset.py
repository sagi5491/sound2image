import torch
import os
import subprocess
from moviepy.editor import VideoFileClip
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from typing import List, Tuple
import json
import librosa
import librosa.display
from PIL import Image
import numpy as np
from torchvision import transforms as transforms
import glob

class MyTransforms:
    def __init__(self) -> None:
        pass

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x

class MyDataset(Dataset):
    def __init__(self, root: str, transform) -> None:
        super().__init__()
        self.transforms = transform
        with open(root + ".json") as f:
            self.data = json.load(f) 
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = Image.open("melspectrograms/" + self.data[index] + ".png")
        label = Image.open("images/" + self.data[index] + ".png")
    
        data = self.transforms(data)
        label = self.transforms(label)

        return data, label

    def __len__(self) -> int:
        return len(self.data)

def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))

def crop_max_square(pil_img):
    return crop_center(pil_img, min(pil_img.size), min(pil_img.size))

def clip_duration(f):
    clip = VideoFileClip(f + ".mp4")
    center = int(clip.duration / 2)
    clip.close()
    return center

def save_img(f, center, width):
    clip = VideoFileClip(f + ".mp4").subclip(center - width, center + width)
    clip.save_frame("images/" + f + ".png", width)
    clip.audio.write_audiofile("audios/" + f + ".wav")
    clip.close()

def to_melspecgram(f):
    y, sr = librosa.load("audios/" + f + ".wav")
    s = librosa.feature.melspectrogram(y=y, sr=sr)
    return librosa.power_to_db(s, ref=np.max)

def save_data(width, size):
    with open("MUSIC21_solo_videos.json") as f:
        di = json.load(f)
    for e in di["videos"]:
        for f in di["videos"][e]:
            url = "https://www.youtube.com/watch?v=" + f
            command = ["yt-dlp.exe", "--force-ipv4", "--quiet", "--no-warnings", "-f", "mp4", "-o", "%s" % (f + ".mp4"), "%s" % url]
            command = " ".join(command)

            attempts = 0
            num_attempts = 5
            status = False
            c = False
            while True:
                try:
                    output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
                except subprocess.CalledProcessError:
                    attempts += 1
                    if attempts == num_attempts:
                        print("error")
                        break
                else:
                    print("downloaded")
                    c = True
                    break
            
            if(c):
                center = clip_duration(f)

                save_img(f, center, width)

                s = to_melspecgram(f)

                load_img = Image.open("images/" + f + ".png")
                trim_img = crop_max_square(load_img)
                reshape_img = trim_img.resize((size, size))
                reshape_img.save("images/" + f + ".png")

                i = (s - np.min(s)) / (np.max(s) - np.min(s))
                img = Image.fromarray(i, "L")
                img.save("melspectrograms/" + f + ".png")

                os.remove(f + ".mp4")

def prepare_dataset(mode):
    transform = transforms.Compose([
        transforms.ToTensor(),
        MyTransforms()
    ])

    if(mode == "train"):

        traindataset = MyDataset(
            root="train",
            transform=transform
        )
        trainloader = DataLoader(
            dataset=traindataset,
            batch_size=64,
            shuffle=True,
            num_workers=2
        )

        return trainloader
    elif(mode == "test"):
        
        testdataset = MyDataset(
            root="test",
            transform=transform
        )
        
        testloader = DataLoader(
            dataset=testdataset,
            batch_size=64,
            shuffle=True,
            num_workers=2
        )
        return testloader

if __name__ == "__main__":
    file_list = glob.glob('images/*.png')
    for name in file_list:
        img = Image.open(name)
        reshape_img = img.resize((127, 127))
        reshape_img.save(name)
