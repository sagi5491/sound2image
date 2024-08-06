import torch
from torchvision import transforms as transforms
from PIL import Image
import model as M

# mel = Image.open("images/bagpipe/kw-OQpF9N4E.png")
# mel = Image.open("melspectrograms/bagpipe/kw-OQpF9N4E.png")
# mel = Image.open("tests/ae_net/test.png")
# mel = Image.open("tests/g_net/test.png")
# mel = transforms.functional.to_tensor(mel)
# print("max: %lf" % torch.max(mel) + " min: %lf" % torch.min(mel))

model = M.D()

input = torch.randn(32, 3, 128, 128)

output = model(input)

print(f"Input shape: {input.shape}")
print(f"Output shape: {output.shape}")
print(f"Output: {output}")