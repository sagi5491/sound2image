import torch
from torchvision import transforms as transforms
import dataset as ds
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ae_test(net):
    print("ae_test")
    criterion = nn.MSELoss()
    net.cuda()
    net.eval()
    criterion.cuda()
    
    losses = []

    test_loader = ds.prepare_dataset("train")
    with torch.no_grad():
        for data, ans, _ in test_loader:
            data = data.to(device)
            ans = ans.to(device)

            _, res = net(data)

            loss = criterion(res, data)

            losses.append(loss.item())

    avg = sum(losses) / len(losses)

    print(avg)

def g_test(ae_net, g_net, d_net):
    print("g_test")
    criterion = nn.MSELoss()
    ae_net.cuda()
    g_net.cuda()
    d_net.cuda()
    ae_net.eval()
    g_net.eval()
    d_net.eval()
    criterion.cuda()

    losses = []

    test_loader = ds.prepare_dataset("train")
    with torch.no_grad():
        for data, ans, _ in test_loader:
            data = data.to(device)
            ans = ans.to(device)

            embeddings, _ = ae_net(data)

            g_out = g_net(embeddings)

            d_out = d_net(g_out, embeddings)

            l = 0.1

            loss = criterion(ans, g_out)

            losses.append(loss.item())

    avg = sum(losses) / len(losses)

    print(avg)

def d_test(ae_net, g_net, d_net):
    print("d_test")
    criterion = nn.MSELoss()
    ae_net.cuda()
    g_net.cuda()
    d_net.cuda()
    ae_net.eval()
    g_net.eval()
    d_net.eval()
    criterion.cuda()

    losses = []

    r_real_tmp = 0
    r_fake_tmp = 0

    test_loader = ds.prepare_dataset("train")
    with torch.no_grad():
        for data, ans, _ in test_loader:
            data = data.to(device)
            ans = ans.to(device)

            embeddings, _ = ae_net(data)

            g_out = g_net(embeddings)

            d_out_real = d_net(ans, embeddings)
            d_out_fake = d_net(g_out, embeddings)

            r_real_tmp += torch.sum(d_out_real) / d_out_real.size(0)
            r_fake_tmp += torch.sum(d_out_fake) / d_out_fake.size(0)

            d_loss = criterion(torch.ones_like(d_out_real), d_out_real) + criterion(torch.full_like(d_out_fake, -1), d_out_fake)

            losses.append(d_loss)
    
    loss = sum(losses) / len(losses)
    r_real = r_real_tmp / len(losses)
    r_fake = r_fake_tmp / len(losses)
        
    print("d_loss: %f" % loss + " r_real: %f" % r_real + " r_fake: %f" % r_fake)

def d_test2(d_net):
    print("d_test2")
    criterion = nn.CrossEntropyLoss()
    d_net.cuda()
    d_net.eval()
    criterion.cuda()

    losses = []

    test_loader = ds.prepare_dataset("test")
    with torch.no_grad():
        for _, ans, label in test_loader:
            ans = ans.to(device)
            label = label.to(device)

            d_out = d_net(ans)

            d_loss = criterion(d_out, label)

            losses.append(d_loss)
    
    loss = sum(losses) / len(losses)
        
    print("d_loss: %f" % loss)