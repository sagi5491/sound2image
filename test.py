import torch
import torch.nn as nn
from torchvision import transforms as transforms

import test_func as Test

if __name__ == "__main__":

    test = input("test(ae_net, g_net, d_net, d_net2): ")
    if(test == "ae_net"):
        ae_model = input("ae_model: ")
        ae_net = torch.load("models/ae_net/" + ae_model)
        Test.ae_test(ae_net)
    elif(test == "g_net"):
        ae_model = input("ae_model: ")
        ae_net = torch.load("models/ae_net/" + ae_model)
        g_model = input("g_model: ")
        g_net = torch.load("models/g_net/" + g_model)
        d_model = input("d_model: ")
        d_net = torch.load("models/d_net/" + d_model)
        Test.g_test(ae_net, g_net, d_net)
    elif(test == "d_net"):
        ae_model = input("ae_model: ")
        ae_net = torch.load("models/ae_net/" + ae_model)
        g_model = input("g_model: ")
        g_net = torch.load("models/g_net/" + g_model)
        d_model = input("d_model: ")
        d_net = torch.load("models/d_net/" + d_model)
        Test.d_test(ae_net, g_net, d_net)
    elif(test == "d_net2"):
        d_model = input("d_model: ")
        d_net = torch.load("models/d_net/" + d_model)
        Test.d_test2(d_net)
    else:
        print("Error: invalid test name")
        exit()
