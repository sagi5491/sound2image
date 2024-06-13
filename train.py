import torch
from torchvision import transforms as transforms
import torch.nn as nn
from torch import optim
import datetime
import dataset as ds
import model as M

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(net, opt, criterion, num_epochs=10):
    net.cuda()
    criterion.cuda()
    net.train()

    for e in range(num_epochs):

        trainloader = ds.prepare_dataset("train")
        print("Epoch: %d" % e)

        for data, ans in trainloader:
            data = data.to(device)
            ans = ans.to(device)

            preds = net(data)

            opt.zero_grad()
            loss = criterion(preds, ans)
            print(loss.item())
            loss.backward()
            opt.step()

def GAN_train(model_g, model_d, num_epochs=10):
    model_G = model_g.to(device)
    model_D = model_d.to(device)

    params_G = optim.Adam(model_G.parameters(), lr=0.001)
    params_D = optim.Adam(model_D.parameters(), lr=0.001)

    true_labels = torch.ones(64, 1, 3, 3).to(device)
    false_labels = torch.zeros(64, 1, 3, 3).to(device)

    bce_loss = nn.BCEWithLogitsLoss()
    mae_loss = nn.L1Loss()

    for i in range(num_epochs):

        trainloader = ds.prepare_dataset("train")
        print("Epoch: %d" % i)

        for data, ans in trainloader:
            batch_len = len(data)

            params_D.zero_grad()
            params_G.zero_grad()

            data = data.to(device)
            ans = ans.to(device)

            fake = model_G(data)
            out = model_D(fake)

            f_t = fake.detach()

            LAMBD = 100.0

            loss_G_bce_tmp = bce_loss(out, true_labels[:batch_len])
            loss_G_mae_tmp = LAMBD * mae_loss(fake, ans)
            loss_G_sum_tmp = loss_G_bce_tmp + loss_G_mae_tmp

            loss_G_sum_tmp.backward()
            params_G.step()

            r = model_D(ans)
            f = model_D(f_t)

            loss_D_real = bce_loss(r, true_labels[:batch_len])
            loss_D_fake = bce_loss(f, false_labels[:batch_len])

            loss_D_tmp = loss_D_real + loss_D_fake

            loss_D_tmp.backward()
            params_D.step()

        print("G: ", loss_G_sum_tmp.item(), " D: ", loss_D_tmp.item())
    
    return model_G, model_D




# if __name__ == "__main__":
#     model = M.Generator()
#     opt = optim.Adam(model.parameters(), lr=0.001)
#     criterion = nn.MSELoss()
#     train(model, opt, criterion, trainloader, num_epochs=10)

#     now = datetime.datetime.now()
#     torch.save(model, "models/model_%s.pth" % now.strftime("%Y%m%d_%H%M%S"))

#     itr = iter(testloader)
#     data, ans = next(itr)
#     data = data.to(device)
#     ans = ans.to(device)

#     model.eval()
#     t = model(data)

#     i = 0
#     for img in t:
#         if i == 0:
#             print(type(img))
#             p = transforms.functional.to_pil_image(img)
#             p.show()
#             i = 1


if __name__ == "__main__":
    model = torch.load("models/model_20240611_145845.pth")
    testloader = ds.prepare_dataset("test")
    itr = iter(testloader)
    data, ans = next(itr)
    data = data.to(device)
    ans = ans.to(device)

    model.eval()
    t = model(data)

    i = 0
    for img in t:
        if i == 0:
            print(type(img))
            
            p = transforms.functional.to_pil_image(img)
            p.show()
            i = 1

# if __name__ == "__main__":
#     model_G, model_D = GAN_train(M.Generator(), M.Discriminator(), 10000)

#     now = datetime.datetime.now()
#     torch.save(model_G, "models/model_%s.pth" % now.strftime("%Y%m%d_%H%M%S"))

#     testloader = ds.prepare_dataset("test")
#     itr = iter(testloader)
#     data, ans = next(itr)
#     data = data.to(device)
#     ans = ans.to(device)

#     model_G.eval()
#     t = model_G(data)

#     i = 0
#     for img in t:
#         if i == 0:
#             print(type(img))
#             p = transforms.functional.to_pil_image(img)
#             p.show()
#             i = 1
