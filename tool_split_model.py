from utils.split_model import *
import torch

def model_construction(dataset, server_layer=0,upload_method=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dataset == "Cifar10":
        model1 = SyNet_client_cifar10().to(device)
        model2 = SyNet_client_cifar10().to(device)
        model3 = SyNet_server_cifar10(upload_method=upload_method).to(device)
        if server_layer == 1:
            model3 = SyNet_server_cifar10().to(device)
        elif server_layer == 2:
            model3 = SyNet_server_cifar10_2().to(device)
        elif server_layer == 3:
            model3 = SyNet_server_cifar10_3().to(device)
        elif server_layer == 4:
            model3 = SyNet_server_cifar10_4().to(device)
        elif server_layer == 5:
            model3 = SyNet_server_cifar10_5().to(device)


    if dataset == "MNIST":
        # Change the model
        model1 = SyNet_client_mnist().to(device)
        model2 = SyNet_client_mnist().to(device)
        model3 = SyNet_server_mnist(upload_method=upload_method).to(device)
        if server_layer == 1:
            model3 = SyNet_server_cifar10().to(device)
        elif server_layer == 2:
            model3 = SyNet_server_cifar10_2().to(device)
        elif server_layer == 3:
            model3 = SyNet_server_cifar10_3().to(device)
        elif server_layer == 4:
            model3 = SyNet_server_cifar10_4().to(device)
        elif server_layer == 5:
            model3 = SyNet_server_cifar10_5().to(device)

        

    
    if dataset == 'imagenette':
        model1 = SyNet_client_imagenette().to(device)
        model2 = SyNet_client_imagenette().to(device)
        model3 = SyNet_server_mnist(upload_method=upload_method).to(device)
        if server_layer == 1:
            model3 = SyNet_server_cifar10().to(device)
        elif server_layer == 2:
            model3 = SyNet_server_cifar10_2().to(device)
        elif server_layer == 3:
            model3 = SyNet_server_cifar10_3().to(device)
        elif server_layer == 4:
            model3 = SyNet_server_cifar10_4().to(device)
        elif server_layer == 5:
            model3 = SyNet_server_cifar10_5().to(device)
    
    if dataset == 'criteo':
        model1 = SyNet_client_criteo().to(device)
        model2 = SyNet_client_criteo().to(device)
        model3 = SyNet_server_criteo(upload_method=upload_method).to(device)

    if dataset == 'cinic10':
        model1 = SyNet_client_cinic10().to(device)
        model2 = SyNet_client_cinic10().to(device)
        model3 = SyNet_server_cinic10(upload_method=upload_method).to(device)
        if server_layer == 1:
            model3 = SyNet_server_cifar10().to(device)
        elif server_layer == 2:
            model3 = SyNet_server_cifar10_2().to(device)
        elif server_layer == 3:
            model3 = SyNet_server_cifar10_3().to(device)
        elif server_layer == 4:
            model3 = SyNet_server_cifar10_4().to(device)
        elif server_layer == 5:
            model3 = SyNet_server_cifar10_5().to(device)

    if dataset == 'givemesomecredit':
        model1 = SyNet_client_credit().to(device)
        model2 = SyNet_client_credit().to(device)
        model3 = SyNet_server_credit(upload_method=upload_method).to(device)
        if server_layer == 1:
            model3 = SyNet_server_bank().to(device)
        elif server_layer == 2:
            model3 = SyNet_server_bank_2().to(device)
        elif server_layer == 3:
            model3 = SyNet_server_bank_3().to(device)
        elif server_layer == 4:
            model3 = SyNet_server_bank_4().to(device)
        elif server_layer == 5:
            model3 = SyNet_server_bank_5().to(device)

    if dataset == 'bank':
        model1 = SyNet_client_bank().to(device)
        model2 = SyNet_client_bank().to(device)
        model3 = SyNet_server_bank(upload_method=upload_method).to(device)
        if server_layer == 1:
            model3 = SyNet_server_bank().to(device)
        elif server_layer == 2:
            model3 = SyNet_server_bank_2().to(device)
        elif server_layer == 3:
            model3 = SyNet_server_bank_3().to(device)
        elif server_layer == 4:
            model3 = SyNet_server_bank_4().to(device)
        elif server_layer == 5:
            model3 = SyNet_server_bank_5().to(device)
            
    if dataset == 'gtsrb':
        model1 = SyNet_client_gtsrb().to(device)
        model2 = SyNet_client_gtsrb().to(device)
        model3 = SyNet_server_gtsrb(upload_method=upload_method).to(device)
        if server_layer == 1:
            model3 = SyNet_server_gtsrb().to(device)
        elif server_layer == 2:
            model3 = SyNet_server_gtsrb_2().to(device)
        elif server_layer == 3:
            model3 = SyNet_server_gtsrb_3().to(device)
        elif server_layer == 4:
            model3 = SyNet_server_gtsrb_4().to(device)
        elif server_layer == 5:
            model3 = SyNet_server_gtsrb_5().to(device)

    return model1, model2, model3
