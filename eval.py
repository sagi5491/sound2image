import torch
import torch.nn as nn
from torchvision import transforms as transforms

import eval_func as Eval

if __name__ == "__main__":
    criterion = nn.MSELoss()

    eval = input("eval(ae_net, g_net, d_net, save_eval): ")
    if(eval == "ae_net"):
        ae_model = input("ae_model: ")
        ae_net = torch.load("models/ae_net/" + ae_model)
        root = input("root: ")
        Eval.ae_eval(ae_net, root)
    elif(eval == "g_net"):
        ae_model = input("ae_model: ")
        ae_net = torch.load("models/ae_net/" + ae_model)
        g_model = input("g_model: ")
        g_net = torch.load("models/g_net/" + g_model)
        root = input("root: ")
        Eval.g_eval(ae_net, g_net, root)
    elif(eval == "d_net"):
        ae_model = input("ae_model: ")
        ae_net = torch.load("models/ae_net/" + ae_model)
        d_model = input("d_model: ")
        d_net = torch.load("models/d_net/" + d_model)
        dataRoot = input("dataRoot: ")
        ansRoot = input("ansRoot: ")
        Eval.d_eval(ae_net, d_net, dataRoot, ansRoot)
    elif(eval == "save_eval"):
        ae_model = input("ae_model: ")
        ae_net = torch.load("models/ae_net/" + ae_model)
        g_model = input("g_model: ")
        g_net = torch.load("models/g_net/" + g_model)
        Eval.save_eval(ae_net, g_net)
    else:
        print("Error: invalid eval name")
        exit()
