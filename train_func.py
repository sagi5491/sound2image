import torch
import dataset as ds

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ae_train(ae_net, criterion, num_epochs=100):
    print("ae_train")
    ae_net.cuda()
    criterion.cuda()
    ae_net.train()
    ae_opt = torch.optim.Adam(ae_net.parameters(), lr=0.001)
    
    ae_hist = []

    for epoch in range(num_epochs):
        print("epoch: %d" % epoch)
        train_loader = ds.prepare_dataset("train")
        for data, ans in train_loader:
            data = data.to(device)

            embeddings, res = ae_net(data)

            ae_opt.zero_grad()
            ae_loss = criterion(res, data)
            
            ae_loss.backward()
            ae_opt.step()

            print("ae_loss: %f " % ae_loss.item())
            ae_hist.append(ae_loss.item())

    return ae_net, ae_hist

def d_train(ae_net, d_net, criterion, num_epochs=100):
    print("d_train")
    ae_net.cuda()
    d_net.cuda()
    criterion.cuda()
    ae_net.eval()
    d_net.train()
    d_opt = torch.optim.Adam(d_net.parameters(), lr=0.0001)

    d_hist = []

    for epoch in range(num_epochs):
        print("epoch: %d" % epoch)
        train_loader = ds.prepare_dataset("train")
        for data, ans in train_loader:
            data = data.to(device)
            ans = ans.to(device)

            embeddings, res = ae_net(data)

            fake = torch.rand(ans.size(0), ans.size(1), ans.size(2), ans.size(3))
            fake = fake * 2 - 1
            fake = fake.to(device)

            # train d_net
            d_out_real = d_net(ans, embeddings)
            d_out_fake = d_net(fake, embeddings)

            r_real = torch.sum(d_out_real) / d_out_real.size(0)
            r_fake = torch.sum(d_out_fake) / d_out_fake.size(0)

            d_loss = criterion(torch.ones_like(d_out_real), d_out_real) + criterion(torch.full_like(d_out_fake, -1), d_out_fake)

            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            
            print("d_loss: %f" % d_loss.item() + " r_real: %f" % r_real + " r_fake: %f" % r_fake)
            d_hist.append(d_loss.item())

    return d_net, d_hist

def train(ae_net, g_net, d_net, criterion, k=2, l=0.1, num_epochs=100):
    print("train")
    ae_net.cuda()
    g_net.cuda()
    d_net.cuda()
    criterion.cuda()
    ae_net.eval()
    g_net.train()
    d_net.train()
    g_opt = torch.optim.Adam(g_net.parameters(), lr=0.001)
    d_opt = torch.optim.Adam(d_net.parameters(), lr=0.0001)

    d_hist = []
    g_hist = []

    for epoch in range(num_epochs):
        print("epoch: %d" % epoch)
        train_loader = ds.prepare_dataset("train")
        count = 0
        for data, ans in train_loader:
            data = data.to(device)
            ans = ans.to(device)

            embeddings, res = ae_net(data)

            g_out = g_net(embeddings)
            

            ans_tensor = ans.detach()
            embeddings_tensor = embeddings.detach()
            g_out_tensor1 = g_out.detach()
            g_out_tensor2 = g_out.detach()

            # train g_net
            d_out = d_net(g_out_tensor1, embeddings)

            g_loss = criterion(ans, g_out) + l * criterion(torch.ones_like(d_out), d_out)
            
            g_opt.zero_grad()
            d_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

            # train d_net
            d_out_real = d_net(ans_tensor, embeddings_tensor)
            d_out_fake = d_net(g_out_tensor2, embeddings_tensor)

            r_real = torch.sum(d_out_real) / d_out_real.size(0)
            r_fake = torch.sum(d_out_fake) / d_out_fake.size(0)

            d_loss = criterion(torch.ones_like(d_out_real), d_out_real) + criterion(torch.full_like(d_out_fake, -1), d_out_fake)

            g_opt.zero_grad()
            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            
            print("g_loss: %f" % g_loss.item() + " d_loss: %f" % d_loss.item() + " r_real: %f" % r_real + " r_fake: %f" % r_fake)
            g_hist.append(g_loss.item())
            d_hist.append(d_loss.item())

            count += 1

    return g_net, d_net, g_hist, d_hist