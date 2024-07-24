import torch
from torchvision import transforms as transforms
from PIL import Image

# mel = Image.open("images/bagpipe/kw-OQpF9N4E.png")
# mel = Image.open("melspectrograms/bagpipe/kw-OQpF9N4E.png")
# mel = Image.open("tests/ae_net/test.png")
mel = Image.open("tests/g_net/test.png")
mel = transforms.functional.to_tensor(mel)
print("max: %lf" % torch.max(mel) + " min: %lf" % torch.min(mel))