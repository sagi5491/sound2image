import torch
from torchvision import transforms as transforms
from PIL import Image
import json
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def ae_eval(net, root):
    print("ae_eval")
    net.cuda()
    net.eval()

    data = Image.open("melspectrograms/" + root + ".png")

    data = transform(data)
    data = torch.unsqueeze(data, 0)

    with torch.no_grad():
        data = data.to(device)

        embeddings, res = net(data)

        img = res[0][0]
        print(img)
        img = (img + 1) / 2
        p = transforms.functional.to_pil_image(img)
        p.save("evals/ae_net/eval.png")

def g_eval(ae_net, g_net, root):
    print("g_eval")
    ae_net.cuda()
    g_net.cuda()
    ae_net.eval()
    g_net.eval()

    data = Image.open("melspectrograms/" + root + ".png")
    ans = Image.open("images/" + root + ".png")

    data = transform(data)
    ans = transform(ans)
    data = torch.unsqueeze(data, 0)
    ans = torch.unsqueeze(ans, 0)

    with torch.no_grad():
        data = data.to(device)
        ans = ans.to(device)

        embeddings, res = ae_net(data)

        output = g_net(embeddings)

        img = output[0]
        img = (img + 1) / 2
        p = transforms.functional.to_pil_image(img)
        p.save("evals/g_net/eval.png")

def d_eval(ae_net, d_net, dataRoot, ansRoot):
    print("d_eval")
    ae_net.cuda()
    d_net.cuda()
    ae_net.eval()
    d_net.eval()

    data = Image.open("melspectrograms/" + dataRoot + ".png")
    ans = Image.open(ansRoot + ".png")

    data = transform(data)
    ans = transform(ans)
    data = torch.unsqueeze(data, 0)
    ans = torch.unsqueeze(ans, 0)

    with torch.no_grad():
        data = data.to(device)
        ans = ans.to(device)

        embeddings, res = ae_net(data)

        d_out = d_net(ans, embeddings)

        r = torch.sum(d_out) / d_out.size(0)
        
    print("r: %f" % r)

def save_eval(ae_net, g_net):
    print("g_eval")
    ae_net.cuda()
    g_net.cuda()
    ae_net.eval()
    g_net.eval()

    with open("test2.json") as f:
        di = json.load(f)

    with torch.no_grad():
        for e in di:

            try:
                os.mkdir("evals/g_net4/" + e)
            except FileExistsError as err:
                print(err)
        
            for f in di[e]:

                try:
                    os.mkdir("evals/g_net4/" + e + "/" + f.split("/")[0])
                except FileExistsError as err:
                    print(err)

                data = Image.open("melspectrograms/" + e + "/" + f + ".png")
                ans = Image.open("images/" + e + "/" + f + ".png")

                data = transform(data)
                ans = transform(ans)

                data = torch.unsqueeze(data, 0)
                ans = torch.unsqueeze(ans, 0)

                data = data.to(device)
                ans = ans.to(device)

                embeddings, res = ae_net(data)

                output = g_net(embeddings)

                img = output[0]
                img = (img + 1) / 2
                p = transforms.functional.to_pil_image(img)
                p.save("evals/g_net4/" + e + "/" + f + ".png")