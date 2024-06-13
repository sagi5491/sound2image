import model as M
import torch

if __name__ == "__main__":
    D_model = M.Discriminator()
    t = torch.rand(1, 3, 127, 127)
    print(t.size())
    r = D_model(t)
    print(r.size())