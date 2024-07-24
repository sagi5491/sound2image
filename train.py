import torch
import torch.nn as nn
from datetime import datetime
import matplotlib.pyplot as plt

import model as M
import train_func as Train

if __name__ == "__main__":
    criterion = nn.MSELoss()
    target = input("target(ae_net, g_net, d_net, gan): ")
    if(target == "ae_net"):
        mode = input("mode(new, pre): ")
        if(mode == "new"):
            ae_net = M.AutoEncoder()

        elif(mode == "pre"):
            ae_model = input("ae_model: ")
            ae_net = torch.load("models/ae_net/" + ae_model)
        
        else:
            print("Error: invalid mode")
            exit()

        num_epochs = input("num_epochs: ")
        ae_net, ae_hist = Train.ae_train(ae_net, criterion, num_epochs=int(num_epochs))
        now = datetime.now()
        torch.save(ae_net, "models/ae_net/ae_%s.pth" % now.strftime("%Y%m%d_%H%M%S"))
        plt.plot(ae_hist, label="ae_loss")
        plt.savefig("losses/ae_net/ae_%s.png" % now.strftime("%Y%m%d_%H%M%S"))

    elif(target == "g_net"):
        mode = input("mode(new, pre): ")
        if(mode == "new"):
            ae_model = input("ae_model: ")
            ae_net = torch.load("models/ae_net/" + ae_model)
            g_net = M.Generator()
            d_model = input("d_model: ")
            d_net = torch.load("models/d_net/" + d_model)
        
        elif(mode == "pre"):
            ae_model = input("ae_model: ")
            ae_net = torch.load("models/ae_net/" + ae_model)
            g_model = input("g_model: ")
            g_net = torch.load("models/g_net/" + g_model)
            d_model = input("d_model: ")
            d_net = torch.load("models/d_net/" + d_model)
        
        else:
            print("Error: invalid mode")
            exit()
        
        num_epochs = input("num_epochs: ")
        g_net, d_net, g_hist, d_hist = Train.train(ae_net, g_net, d_net, criterion, num_epochs=int(num_epochs))
        now = datetime.now()
        torch.save(g_net, "models/g_net/g_%s.pth" % now.strftime("%Y%m%d_%H%M%S"))
        plt.plot(g_hist, label="g_loss")
        plt.savefig("losses/g_net/g_%s.png" % now.strftime("%Y%m%d_%H%M%S"))
    
    elif(target == "d_net"):
        mode = input("mode(new, pre): ")
        if(mode == "new"):
            ae_model = input("ae_model: ")
            ae_net = torch.load("models/ae_net/" + ae_model)
            d_net = M.Discriminator()
        
        elif(mode == "pre"):
            ae_model = input("ae_model: ")
            ae_net = torch.load("models/ae_net/" + ae_model)
            d_model = input("d_model: ")
            d_net = torch.load("models/d_net/" + d_model)
        
        else:
            print("Error: invalid mode")
            exit()
        
        num_epochs = input("num_epochs: ")
        d_net, d_hist = Train.d_train(ae_net, d_net, criterion, num_epochs=int(num_epochs))
        now = datetime.now()
        torch.save(d_net, "models/d_net/d_%s.pth" % now.strftime("%Y%m%d_%H%M%S"))
        plt.plot(d_hist, label="d_loss")
        plt.savefig("losses/d_net/d_%s.png" % now.strftime("%Y%m%d_%H%M%S"))

    elif(target == "gan"):
        mode = input("mode(new, g_pre, d_pre, pre): ")
        if(mode == "new"):
            ae_model = input("ae_model: ")
            ae_net = torch.load("models/ae_net/" + ae_model)
            g_net = M.Generator()
            d_net = M.Discriminator()

        elif(mode == "g_pre"):
            ae_model = input("ae_model: ")
            ae_net = torch.load("models/ae_net/" + ae_model)
            g_model = input("g_model: ")
            g_net = torch.load("models/g_net/" + g_model)
            d_net = M.Discriminator()

        elif(mode == "d_pre"):
            ae_model = input("ae_model: ")
            ae_net = torch.load("models/ae_net/" + ae_model)
            g_net = M.Generator()
            d_model = input("d_model: ")
            d_net = torch.load("models/d_net/" + d_model)

        elif(mode == "pre"):
            ae_model = input("ae_model: ")
            ae_net = torch.load("models/ae_net/" + ae_model)
            g_model = input("g_model: ")
            g_net = torch.load("models/g_net/" + g_model)
            d_model = input("d_model: ")
            d_net = torch.load("models/d_net/" + d_model)

        else:
            print("Error: invalid mode")
            exit()

        num_epochs = input("num_epochs: ")
        g_net, d_net, g_hist, d_hist = Train.train(ae_net, g_net, d_net, criterion, num_epochs=int(num_epochs))
        now = datetime.now()
        torch.save(g_net, "models/g_net/g_%s.pth" % now.strftime("%Y%m%d_%H%M%S"))
        torch.save(d_net, "models/d_net/d_%s.pth" % now.strftime("%Y%m%d_%H%M%S"))
        plt.plot(g_hist, label="g_loss")
        plt.plot(d_hist, label="d_loss")
        plt.savefig("losses/gan/gan_%s.png" % now.strftime("%Y%m%d_%H%M%S"))

    else:
        print("Error: invalid target")
        exit()

