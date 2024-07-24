import torch
from torchvision import transforms as transforms
import dataset as ds

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ae_test(net, criterion):
    print("ae_test")
    net.cuda()
    net.eval()
    criterion.cuda()
    
    losses = []

    test_loader = ds.prepare_dataset("train")
    with torch.no_grad():
        for data, ans in test_loader:
            data = data.to(device)
            ans = ans.to(device)

            embeddings, res = net(data)

            loss = criterion(res, data)

            losses.append(loss.item())

    avg = sum(losses) / len(losses)

    print(avg)

def g_test(ae_net, g_net, d_net, criterion):
    print("g_test")
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
        for data, ans in test_loader:
            data = data.to(device)
            ans = ans.to(device)

            embeddings, res = ae_net(data)

            g_out = g_net(embeddings)

            d_out = d_net(g_out, embeddings)

            l = 0.1

            loss = criterion(ans, g_out)

            losses.append(loss.item())

    avg = sum(losses) / len(losses)

    print(avg)

def d_test(ae_net, g_net, d_net, criterion):
    print("d_test")
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
    size = 0

    test_loader = ds.prepare_dataset("train")
    with torch.no_grad():
        for data, ans in test_loader:
            data = data.to(device)
            ans = ans.to(device)

            embeddings, res = ae_net(data)

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