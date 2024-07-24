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
import matplotlib.pyplot as plt
import soundfile as sf

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
            di = json.load(f)
        self.path = []
        for e in di:
            for f in di[e]:
                self.path.append(e + "/" + f)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = Image.open("melspectrograms/" + self.path[index] + ".png")
        ans = Image.open("images/" + self.path[index] + ".png")
    
        data = self.transforms(data)
        ans = self.transforms(ans)

        return data, ans

    def __len__(self) -> int:
        return len(self.path)

def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))

def crop_max_square(pil_img):
    return crop_center(pil_img, min(pil_img.size), min(pil_img.size))

def clip_duration(f, width):
    clip = VideoFileClip(f + ".mp4")
    len = int((clip.duration - 2 * width - 1) / 20)
    clip.close()
    return len

def save_img(filename, img_root, audio_root, center, width):
    clip = VideoFileClip(filename + ".mp4").subclip(center - width, center + width)
    clip.save_frame(img_root, width)
    clip.audio.write_audiofile(audio_root)
    clip.close()

def save_data(width, size):
    with open("MUSIC21_solo_videos.json") as f:
        di = json.load(f)
    status = False
    for e in di["videos"]:

        try:
            os.mkdir("audios/" + e)
        except FileExistsError as err:
            print(err)
        try:
            os.mkdir("images/" + e)
        except FileExistsError as err:
            print(err)
        try:
            os.mkdir("melspectrograms/" + e)
        except FileExistsError as err:
            print(err)
        try:
            os.mkdir("audios_from_mel/" + e)
        except FileExistsError as err:
            print(err)
        try:
            os.mkdir("audios_from_img/" + e)
        except FileExistsError as err:
            print(err)
        try:
            os.mkdir("mel_from_mel/" + e)
        except FileExistsError as err:
            print(err)
        try:
            os.mkdir("mel_from_img/" + e)
        except FileExistsError as err:
            print(err)
        
        for f in di["videos"][e]:

            if(f == "CeOAuSm1NUo"):
                status = True
            elif(status == False):
                continue

            url = "https://www.youtube.com/watch?v=" + f
            command = ["yt-dlp.exe", "--force-ipv4", "--quiet", "--no-warnings", "-f", "mp4", "-o", "%s" % (f + ".mp4"), "%s" % url]
            command = " ".join(command)

            attempts = 0
            num_attempts = 5
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

                
                center = width + 0.5
                len = clip_duration(f, width)

                try:
                    os.mkdir("audios/" + e + "/" + f)
                except FileExistsError as err:
                    print(err)
                try:
                    os.mkdir("images/" + e + "/" + f)
                except FileExistsError as err:
                    print(err)
                try:
                    os.mkdir("melspectrograms/" + e + "/" + f)
                except FileExistsError as err:
                    print(err)
                try:
                    os.mkdir("audios_from_mel/" + e + "/" + f)
                except FileExistsError as err:
                    print(err)
                try:
                    os.mkdir("audios_from_img/" + e + "/" + f)
                except FileExistsError as err:
                    print(err)
                try:
                    os.mkdir("mel_from_mel/" + e + "/" + f)
                except FileExistsError as err:
                    print(err)
                try:
                    os.mkdir("mel_from_img/" + e + "/" + f)
                except FileExistsError as err:
                    print(err)
                
                for num in range(20):

                    # 画像抽出
                    center += len
                    save_img(f, "images/" + e + "/" + f + "/" + str(num) + ".png", "audios/" + e + "/" + f + "/" + str(num) + ".wav", center, width)
                    img = Image.open("images/" + e + "/" + f + "/" + str(num) + ".png")
                    img = crop_center(img, 128, 128)
                    img.save("images/" + e + "/" + f + "/" + str(num) + ".png")

                    # メルスペクトログラムを直に戻した際の音
                    y, sr = librosa.load("audios/" + e + "/" + f + "/" + str(num) + ".wav")
                    s = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, win_length=512, hop_length=512)
                    y = librosa.feature.inverse.mel_to_audio(s)
                    sf.write("audios_from_mel/" + e + "/" + f + "/" + str(num) + ".wav", y, 22050, format='WAV', subtype='PCM_16')

                    # メルスペクトログラム
                    s_db = librosa.power_to_db(s, ref=np.max)
                    i = librosa.display.specshow(s_db, sr=sr)
                    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
                    plt.savefig("melspectrograms/" + e + "/" + f + "/" + str(num) + ".png")
                    mel = Image.open("melspectrograms/" + e + "/" + f + "/" + str(num) + ".png").convert('L')
                    mel = mel.resize((128, 128))
                    mel.save("melspectrograms/" + e + "/" + f + "/" + str(num) + ".png")

                    # 処理を施したメルスペクトログラムを戻した際の音
                    mel = Image.open("melspectrograms/" + e + "/" + f + "/" + str(num) + ".png").convert('L')
                    mel = np.array(mel)
                    mel = mel.astype(np.float32)
                    mel = (mel / 255.0) * 80.0 - 80.0
                    S_dB = mel
                    S = librosa.db_to_power(S_dB, ref=1.0)
                    y = librosa.feature.inverse.mel_to_audio(S, sr=22050, n_fft=2048, hop_length=512)
                    sf.write("audios_from_img/" + e + "/" + f + "/" + str(num) + ".wav", y, 22050, format='WAV', subtype='PCM_16')

                    # メルスペクトログラムを直に戻した際の音からさらにメルスペクトログラムを作成する
                    y, sr = librosa.load("audios_from_mel/" + e + "/" + f + "/" + str(num) + ".wav")
                    s = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, win_length=512, hop_length=512)
                    s_db = librosa.power_to_db(s, ref=np.max)
                    i = librosa.display.specshow(s_db, sr=sr)
                    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
                    plt.savefig("mel_from_mel/" + e + "/" + f + "/" + str(num) + ".png")

                    # 処理を施したメルスペクトログラムを戻した際の音からさらにメルスペクトログラムを作成する
                    y, sr = librosa.load("audios_from_img/" + e + "/" + f + "/" + str(num) + ".wav")
                    s = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, win_length=512, hop_length=512)
                    s_db = librosa.power_to_db(s, ref=np.max)
                    i = librosa.display.specshow(s_db, sr=sr)
                    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
                    plt.savefig("mel_from_img/" + e + "/" + f + "/" + str(num) + ".png")

                os.remove(f + ".mp4")

def prepare_dataset(mode):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if(mode == "train"):

        traindataset = MyDataset(
            root="train2",
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
            root="test2",
            transform=transform
        )
        
        testloader = DataLoader(
            dataset=testdataset,
            batch_size=32,
            shuffle=False,
            num_workers=2
        )
        return testloader

# if __name__ == "__main__":
#     file_list = glob.glob('images/*.png')
#     for name in file_list:
#         img = Image.open(name)
#         reshape_img = img.resize((127, 127))
#         reshape_img.save(name)

if __name__ == "__main__":
    save_data(1.485, 128)