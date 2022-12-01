#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def param():
    description = "split information"    
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset", type=str, default="Cifar10")
    parser.add_argument("--target_label", type=int, default=0)
    parser.add_argument("--target_label_candidates_number", type=int, default=1)
    parser.add_argument("--noise_range", type=float, default=0.03)
    parser.add_argument("--noise_p", type=float, default=0.5)
    parser.add_argument("--mask_size", type=int, default=64)
    parser.add_argument("--poisoning_rate", type=float, default=0.01)
    parser.add_argument("--wave_multiple", type=float, default=0.4)
    parser.add_argument("--backdoor_epochs", type=int, default=60)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--id_attacker", type=int, default=1)
    args = parser.parse_args()
    return args


# In[ ]:


import argparse

args = param()

target_dataset = args.dataset
target_label = args.target_label
target_label_candidates_number = args.target_label_candidates_number
noise_range = args.noise_range
noise_p = args.noise_p
mask_size = args.mask_size
poisoning_rate = args.poisoning_rate
wave_multiple = args.wave_multiple
backdoor_epochs = args.backdoor_epochs
n_workers = args.n_workers
id_attacker = args.id_attacker
model_type = '_attacker_'+str(id_attacker)+'_normal'

# In[152]:


from uuid import uuid4
from torch import nn
import numpy as np
import torch
import tqdm
from torchvision import datasets
from torchvision import transforms
import random
from random import sample
import copy
from simple_colors import *
from torch.utils.data.sampler import WeightedRandomSampler

from matplotlib import pyplot as plt
from utils.split_model_split import *
import dataset

import datetime
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Results_path = "./Results/onequery/" + target_dataset + "/"

if target_dataset == "Cifar10":
    epochs = 30
    batch_size = 128
    search_N = 14
    labels_kind = 10

if target_dataset == "MNIST":
    epochs = 10
    batch_size = 128
    search_N = 14
    labels_kind = 10

if target_dataset == 'imagenette':
    epochs = 30
    batch_size = 50
    search_N = 10
    labels_kind = 10

if target_dataset == 'cinic10':
    epochs = 30
    batch_size = 64
    search_N = 8
    labels_kind = 10

if target_dataset == 'givemesomecredit':
    epochs = 5
    batch_size = 1000
    search_N = 40
    labels_kind = 2

if target_dataset == 'bank':
    epochs = 10
    batch_size = 100
    search_N = 6
    labels_kind = 2


# In[153]:


print("Loading dataset...")
split_train_dataset, split_val_dataset, train_labels, val_labels = dataset.get_train_test_set(
    target_dataset,n_workers=n_workers)

train_num = split_train_dataset[0].shape[0]
val_num = split_val_dataset[0].shape[0]

imgs = split_train_dataset
val_imgs = split_val_dataset

labels = train_labels
labels_val = val_labels

def shuffle_data(imgs, label):
    ll = []
    for i in range(len(label)):
        group = []
        for element in imgs:
            group.append(element[i])
        ll.append((group, labels[i]))
    random.shuffle(ll)
    imgs, label = zip(*ll)
    imgs = list(zip(*imgs))
    return imgs, label

all_imgs, all_labels = shuffle_data(imgs,labels)

train_data = []
val_data = []
for i in range(n_workers):
    train_data.append(torch.utils.data.DataLoader(all_imgs[i],batch_size=batch_size))
    val_data.append(torch.utils.data.DataLoader(val_imgs[i],batch_size=batch_size))

labels = torch.utils.data.DataLoader(all_labels, batch_size=batch_size)
val_labels = torch.utils.data.DataLoader(labels_val, batch_size=batch_size)
print(f"Finish loading {target_dataset} dataset")


# In[120]:


# 保留中间变量函数
grads = {}

def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook

# In[154]:


model_bottom = []
opt_bottom = []
criterion = nn.CrossEntropyLoss()
if target_dataset == "Cifar10":
    for i in range(n_workers):
        model_bottom.append(SyNet_client_cifar10(n_workers=n_workers).to(device))
    model_top = SyNet_server_cifar10().to(device)
elif target_dataset == "gtsrb":
    for i in range(n_workers):
        model_bottom.append(SyNet_client_gtsrb(n_workers=n_workers).to(device))
    model_top = SyNet_server_gtsrb().to(device)
elif target_dataset == "MNIST":
    for i in range(n_workers):
        model_bottom.append(SyNet_client_mnist(n_workers=n_workers).to(device))
    model_top = SyNet_server_mnist().to(device)
elif target_dataset == "bank":
    for i in range(n_workers):
        model_bottom.append(SyNet_client_bank(n_workers=n_workers).to(device))
    model_top = SyNet_server_bank().to(device)
elif target_dataset == "givemesomecredit":
    for i in range(n_workers):
        model_bottom.append(SyNet_client_credit(n_workers=n_workers).to(device))
    model_top = SyNet_server_credit().to(device)
elif target_dataset == "cinic10":
    for i in range(n_workers):
        model_bottom.append(SyNet_client_cinic10(n_workers=n_workers).to(device))
    model_top = SyNet_server_cinic10().to(device)
elif target_dataset == "imagenette":
    for i in range(n_workers):
        model_bottom.append(SyNet_client_imagenette(n_workers=n_workers).to(device))
    model_top = SyNet_server_imagenette().to(device)

if target_dataset == 'imagenette':
    for i in range(n_workers):
        if i == id_attacker:
            opt_bottom.append(torch.optim.SGD(params=model_bottom[i].parameters(),lr=0.005))
        else:
            opt_bottom.append(torch.optim.SGD(params=model_bottom[i].parameters(),lr=0.0005))
    opt_top = torch.optim.SGD(params=model_top.parameters(),lr=0.01)
else:
    for i in range(n_workers):
        if i == id_attacker:
            opt_bottom.append(torch.optim.SGD(params=model_bottom[i].parameters(),lr=0.1))
        else:
            opt_bottom.append(torch.optim.SGD(params=model_bottom[i].parameters(),lr=0.01))
    opt_top = torch.optim.SGD(params=model_top.parameters(),lr=0.15)


# In[155]:


def get_embedding_len(model_bottom, model_top, train_data, labels, n_workers):
    dl = list(zip(*train_data))
    embedding_len = 0
    for model in model_bottom:
        model.eval()
    model_top.eval()
    for input,label in zip(dl,labels):
        output = []
        for i in range(n_workers):
            inp = input[i]
            inp = inp.to(device)
            if target_dataset == 'bank' or target_dataset == 'givemesomecredit':
                inp = inp.float()
            if i == id_attacker and embedding_len == 0:
                a1 = model_bottom[i](inp)
                embedding_len = len(a1[0])
                break
        if embedding_len != 0:
            break
    return embedding_len


# In[122]:


def backdoor_stripe(tensor, wave_amp, scope=0, mask=None):
    if scope == 0:
        if wave_amp <0.15:
            wave_amp = 0.15
            #print('wave_amp:', wave_amp)
        tensor_len = len(tensor)
        tensor_multiple = int(tensor_len/4)
        for i in range(tensor_multiple):
            if int(mask[i*4 + 0]) == 1:
                tensor[i*4 + 0] = tensor[i*4 + 0] + wave_amp
            if int(mask[i*4 + 1]) == 1:
                tensor[i*4 + 1] = tensor[i*4 + 1] + wave_amp
            if int(mask[i*4 + 2]) == 1:
                tensor[i*4 + 2] = tensor[i*4 + 2] - wave_amp
            if int(mask[i*4 + 3]) == 1:
                tensor[i*4 + 3] = tensor[i*4 + 3] - wave_amp
        for i in range(tensor_multiple*4, tensor_len):
            if (i % 4 == 0 or i % 4 == 1) and int(mask[i]) == 1:
                tensor[i*4 + 0] = tensor[i*4 + 0] + wave_amp
            if (i % 4 == 2 or i % 4 == 3) and int(mask[i]) == 1:
                tensor[i*4 + 0] = tensor[i*4 + 0] - wave_amp
    elif scope == 1:
        for i in range(1):
            tensor[i*4 + 60] = tensor[i*4 + 60] + wave_amp
            tensor[i*4 + 61] = tensor[i*4 + 61] + wave_amp
            tensor[i*4 + 62] = tensor[i*4 + 62] - wave_amp
            tensor[i*4 + 63] = tensor[i*4 + 63] - wave_amp
    elif scope == 2:
        for i in range(2):
            tensor[i*4 + 56] = tensor[i*4 + 56] + wave_amp
            tensor[i*4 + 57] = tensor[i*4 + 57] + wave_amp
            tensor[i*4 + 58] = tensor[i*4 + 58] - wave_amp
            tensor[i*4 + 59] = tensor[i*4 + 59] - wave_amp
    elif scope == 3:
        for i in range(4):
            tensor[i*4 + 48] = tensor[i*4 + 48] + wave_amp
            tensor[i*4 + 49] = tensor[i*4 + 49] + wave_amp
            tensor[i*4 + 50] = tensor[i*4 + 50] - wave_amp
            tensor[i*4 + 51] = tensor[i*4 + 51] - wave_amp
    elif scope == 4:
        for i in range(8):
            tensor[i*4 + 32] = tensor[i*4 + 32] + wave_amp
            tensor[i*4 + 33] = tensor[i*4 + 33] + wave_amp
            tensor[i*4 + 34] = tensor[i*4 + 34] - wave_amp
            tensor[i*4 + 35] = tensor[i*4 + 35] - wave_amp
    return tensor


# In[123]:


def all_embedding_statistics(model_bottom, model_top, opt_bottom, opt_top, train_data, labels, criterion, batch_size, device, train_num, labels_kind, n_workers=2):
    
    dl = list(zip(*train_data))
    iternum = 0
    embedding_list = []
    label_list = []

    for model in model_bottom:
        model.eval()
    model_top.eval()
    for input,label in zip(dl,labels):
        output = []
        # input data process
        opt_top.zero_grad()
        for i in range(n_workers):
            opt_bottom[i].zero_grad()
        label = label.to(device)
        if target_dataset == 'bank' or target_dataset == 'givemesomecredit':
            label = label.float()
        # output of models
        for i in range(n_workers):
            inp = input[i]
            inp = inp.to(device)
            if target_dataset == 'bank' or target_dataset == 'givemesomecredit':
                inp = inp.float()
            if i == id_attacker:
                a1 = model_bottom[i](inp)
                output.append(a1)
            else:
                output.append(model_bottom[i](inp))
        a = torch.cat(output,1)
        y = model_top(a)
        loss = criterion(y, label.long())
        loss.backward()
        embedding_list.append(a1)
        label_list.append(label)
        iternum += 1

    #for i in range(labels_kind+1):
    for i in range(1):
        embedding_mean = torch.zeros_like(embedding_list[0][0])
        embedding_std = torch.zeros_like(embedding_list[0][0])
        if i == 0:
            print("All Embedding:")
        else:
            print("Embedding with label ",i-1,":")
        for num_in_embedding in range(len(embedding_list[0][0])):
            norm_list = []
            if i == 0:
                for j in range(len(embedding_list)):
                    for k in range(len(embedding_list[j])):
                        norm_list.append(float(embedding_list[j][k][num_in_embedding]))
            else:
                for j in range(len(embedding_list)):
                    for k in range(len(embedding_list[j])):
                        if label_list[j][k] == i-1:
                            norm_list.append(float(embedding_list[j][k][num_in_embedding]))
            mean = np.mean(norm_list)
            std = np.std(norm_list)
            embedding_mean[num_in_embedding] = mean
            embedding_std[num_in_embedding] = std
        print("Mean Embedding:")
        print(embedding_mean)
        print("Std Embedding:")
        print(embedding_std)
    return embedding_mean, embedding_std 


# In[124]:


def model_training(model_bottom, model_top, opt_bottom, opt_top, train_data, labels, val_data, val_labels, epochs, device, criterion, n_workers=2):
    train_acc_list = []
    train_loss_list = []
    val_acc_list = []
    val_loss_list = []
    dl = list(zip(*train_data))
    val_dl = list(zip(*val_data))
    embedding_len = 0

    for epoch in range(epochs):
        train_correct = 0
        train_total = 0
        train_loss_epoch = 0
        val_loss_epoch = 0
        for model in model_bottom:
            model.train()
        model_top.train()
        for input,label in zip(dl,labels):
            output = []
            opt_top.zero_grad()
            for i in range(n_workers):
                opt_bottom[i].zero_grad()
                inp = input[i]
                inp = inp.to(device)
                if target_dataset == 'bank' or target_dataset == 'givemesomecredit':
                    inp = inp.float()
                if i == id_attacker and embedding_len == 0:
                    a1 = model_bottom[i](inp)
                    embedding_len = len(a1[0])
                output.append(model_bottom[i](inp))
            a = torch.cat(output,1)
            label = label.to(device)
            if target_dataset == 'bank' or target_dataset == 'givemesomecredit':
                label = label.float()
            y = model_top(a)
            loss = criterion(y, label.long())
            train_correct += y.max(1)[1].eq(label).sum().item()
            train_total += y.shape[0]
            loss.backward()
            opt_top.step()
            for opt in opt_bottom:
                opt.step()
            train_loss_epoch += loss.item()*a.size(0)
        train_loss = train_loss_epoch/train_total
        correct = 0
        total = 0
        for model in model_bottom:
            model.eval()
        model_top.eval()
        for input,label in zip(val_dl,val_labels):
            output = []
            for i in range(n_workers):
                inp = input[i]
                inp = inp.to(device)
                if target_dataset == 'bank' or target_dataset == 'givemesomecredit':
                    inp = inp.float()
                output.append(model_bottom[i](inp))
            a = torch.cat(output,1)
            label = label.to(device)
            if target_dataset == 'bank' or target_dataset == 'givemesomecredit':
                label = label.float()
            y = model_top(a)
            loss = criterion(y,label.long())
            val_loss_epoch+= loss.item()*a.size(0)
            correct += y.max(1)[1].eq(label).sum().item()
            total += y.shape[0]
        val_loss = val_loss_epoch/total
        print(f"Current epoch {epoch+1}/{epochs}...")
        print(f"Train Accuracy: {100*train_correct/train_total:.3f}%")
        print(f"Current train loss: {train_loss}")
        print(f"Val Accuracy: {100*correct/total:.3f}%")
        print(f"Val loss: {val_loss}")
        train_acc_list.append(train_correct/train_total)
        train_loss_list.append(train_loss)
        val_acc_list.append(correct/total)
        val_loss_list.append(val_loss)
        
    return model_bottom, model_top, embedding_len

# get correct and gradient list

def correctness_gradient_list(model_bottom, model_top, train_data, labels, criterion, batch_size, device, train_num, n_workers=2, embedding_len=64):
    dl = list(zip(*train_data))
    iternum = 0
    train_dataset_correct_list = []
    train_dataset_gradient = torch.zeros(train_num)
    train_dataset_gradient_comp = torch.zeros(train_num, embedding_len)

    for model in model_bottom:
        model.eval()
    model_top.eval()
    for input,label in zip(dl,labels):
        output = []
        # input data process
        opt_top.zero_grad()
        for i in range(n_workers):
            opt_bottom[i].zero_grad()
        label = label.to(device)
        if target_dataset == 'bank' or target_dataset == 'givemesomecredit':
            label = label.float()
        # output of bottom models
        for i in range(n_workers):
            inp = input[i]
            inp = inp.to(device)
            if target_dataset == 'bank' or target_dataset == 'givemesomecredit':
                inp = inp.float()
            if i == id_attacker:
                a1 = model_bottom[i](inp)
                output.append(a1)
            else:
                output.append(model_bottom[i](inp))
        a = torch.cat(output,1)
        y = model_top(a)
        h = a1.register_hook(save_grad('a1'))
        loss = criterion(y, label.long())
        loss.backward()
        for i in range(len(label)):
            train_dataset_gradient[i + batch_size*iternum] = torch.norm(grads['a1'][i]).to('cpu')
            train_dataset_gradient_comp[i + batch_size*iternum] = grads['a1'][i].to('cpu')
            if torch.argmax(y[i]) == label[i]:
                train_dataset_correct_list.append(i + batch_size*iternum)
        iternum += 1
    h.remove()
    return train_dataset_correct_list, train_dataset_gradient, train_dataset_gradient_comp

# get the embedding id and embedding

def embedding_random_choose(target_label, target_label_candidates_number, all_labels, train_dataset_correct_list, train_dataset_gradient, model_bottom, all_imgs, device='cuda', n_workers=2):
    img0 = all_imgs[id_attacker]
    model1 = model_bottom[id_attacker]
    target_label_list = []
    target_label_candidates = []

    for i in range(len(all_labels)):
        if i in train_dataset_correct_list and int(all_labels[i].item()) == target_label:
            target_label_list.append(i)
    if target_label_candidates_number > len(target_label_list):
        target_label_candidates = random.sample(target_label_list, len(target_label_list))
    else:
        target_label_candidates = random.sample(target_label_list, target_label_candidates_number)
    print(target_label_candidates)

    a = [train_dataset_gradient[i] for i in target_label_candidates]
    print(a)
    target_label_embedding_id = target_label_candidates[a.index(min(a))]

    a1 = model1(img0[target_label_embedding_id].unsqueeze(0).to(device))
    target_label_embedding = a1

    return target_label_embedding_id, target_label_embedding

# calculate the attack successful rate with direct replacement

def direct_replacement_successful_rate(model_bottom, model_top, val_data, val_labels, embedding, target_label, device='cuda', n_workers=2):
    val_dl = list(zip(*val_data))
    correct = 0
    total = 0
    for model in model_bottom:
        model.eval()
    model_top.eval()
    for input,label in zip(val_dl,val_labels):
        output = []
        # input data process
        opt_top.zero_grad()
        for i in range(n_workers):
            opt_bottom[i].zero_grad()
        label = label.to(device)
        if target_dataset == 'bank' or target_dataset == 'givemesomecredit':
            label = label.float()
        # output of bottom models
        for i in range(n_workers):
            inp = input[i]
            inp = inp.to(device)
            if target_dataset == 'bank' or target_dataset == 'givemesomecredit':
                inp = inp.float()
            if i == id_attacker:
                a1 = model_bottom[i](inp)
                for j in range(len(label)):
                    a1.data[j] = embedding[0]
                    label[j] = target_label
                output.append(a1)
            else:
                output.append(model_bottom[i](inp))
        a = torch.cat(output,1)
        y = model_top(a)
        correct += y.max(1)[1].eq(label).sum().item()
        total += y.shape[0]
    print(f"Backdoor Success Rate: {100*correct/total:.3f}%")

# label inference

def label_inference(model_bottom, model_top, embedding_id, embedding, train_data, labels, target_dataset, target_label, labels_kind, search_N, batch_size, device, random_samples_flag=0, n_workers=2, embedding_len=64):
    dl = list(zip(*train_data))
    M_D1 = M_Discriminator(embedding_len).to(device)

    # inference
    # target_label = 2
    D_id = [embedding_id]
    D_label = torch.tensor([1]).to(device)
    D_embedding = embedding.squeeze(0).clone()
    D_label = D_label.detach()
    D_embedding = D_embedding.unsqueeze(0).detach()

    # dynamic embedding
    embedding_num = 20
    embedding_count = 1
    embedding_library = torch.Tensor(np.ones([embedding_num, embedding_len])).to(device)
    embedding_library[0] = embedding

    # number of selected ids in a batchsize
    N = search_N
    # batch counter
    id = 0

    adversarial_loss = torch.nn.BCELoss()
    local_opt_M = torch.optim.SGD(params=M_D1.parameters(),lr=0.1)
    
    #targeted label inference attack
    all_target = 0
    total_target = 0
    true_target = 0
    begin = 0

    if target_dataset == 'imagenette':
        local_opt_M = torch.optim.Adam(params=M_D1.parameters(), lr=0.01)

    for input,label in zip(dl,labels):
        # input data process
        label = label.to(device)
        if target_dataset == 'bank' or target_dataset == 'givemesomecredit':
            label = label.float()
        
        for i in range(len(label)):
            if label[i] == target_label:
                all_target += 1

        if (begin == 1) and (random_samples_flag == 0):
            # train the model M with D_embedding and D_label
            M_D1_label = torch.tensor(list(map(lambda x: 1.0 if x == target_label else 0.0, D_label))).to(device)

            # balance the data
            weights = [M_D1_label.tolist().count(0.0)/M_D1_label.tolist().count(1.0) if label == 1.0 else 1 for label in M_D1_label]
            M_sampler = WeightedRandomSampler(weights,num_samples=len(weights)*N, replacement=True)

            M_dataset = torch.cat((D_embedding, M_D1_label.unsqueeze(1)) ,1)

            M_D1_data = torch.utils.data.DataLoader(M_dataset, batch_size=10*N, sampler=M_sampler)
            M_D1_datalabel = torch.utils.data.DataLoader(M_D1_label, batch_size=N)
            
            for M_data, in zip(M_D1_data):
                if target_dataset == 'criteo' and id > 100:
                    break
                M_in = M_data[:, 0:embedding_len]
                M_label = M_data[:, embedding_len].squeeze()
                local_opt_M.zero_grad()
                output = M_D1(M_in).squeeze()
                loss_M = adversarial_loss(output, M_label)
                loss_M.backward()
                local_opt_M.step()

            # select N pieces of data in a batch
            # change the selected function to get the most target label data with model M
            inp = input[id_attacker]
            inp = inp.to(device)
            if target_dataset == 'bank' or target_dataset == 'givemesomecredit':
                inp = inp.float()
            a1 = model_bottom[id_attacker](inp)
            probability = M_D1(a1)
            if target_dataset == 'bank' and inp.shape[0] < batch_size:
                break
            if len(label) > N:
                _N = N
            else:
                _N = len(label)
            topN_indices = torch.topk(probability.squeeze(), _N).indices
            label_topN = label[topN_indices]
            id_selected_in_batch = topN_indices.tolist()

        else:
            _N = N
            if len(label) < _N:
                id_selected_in_batch = random.sample(range(len(label)), len(label))
            else:
                id_selected_in_batch = random.sample(range(len(label)), _N)
            begin = 1

        numpy_tensor=np.ones([_N, 3])
        g_norm=torch.Tensor(numpy_tensor).to(device)

        for i in range(2):
            # Loop for every embedding
            for model in model_bottom:
                model.eval()
            model_top.eval()

            # output of bottom models
            output = []
            for j in range(n_workers):
                inp = input[j]
                inp = inp.to(device)
                if target_dataset == 'bank' or target_dataset == 'givemesomecredit':
                    inp = inp.float()
                if j == id_attacker:
                    a1 = model_bottom[j](inp)
                    if i == 1:
                        for k in id_selected_in_batch:
                            a1.data[k] = embedding
                    output.append(a1)
                else:
                    output.append(model_bottom[j](inp))

            a = torch.cat(output,1)
            y = model_top(a)

            h_a1 = a1.register_hook(save_grad('a1'))
            h_y = y.register_hook(save_grad('y'))

            criterion = nn.NLLLoss()
            loss = criterion(y, label.long())       
            loss.backward()

            for i_id_selected_in_batch in range(len(id_selected_in_batch)):
                g_norm[i_id_selected_in_batch, i] = torch.norm(grads['a1'][id_selected_in_batch[i_id_selected_in_batch]]).to('cpu')

        for i_id_selected_in_batch in range(len(id_selected_in_batch)):
            if g_norm[i_id_selected_in_batch, 1]/g_norm[i_id_selected_in_batch, 0] < 5 and g_norm[i_id_selected_in_batch, 0] < 0.01:
                total_target += 1
                g_norm[i_id_selected_in_batch, 2] = target_label 
                if (embedding_count < embedding_num) and (g_norm[i_id_selected_in_batch, 1] < 1e-6):
                    inp = input[id_attacker]
                    inp = inp.to(device)
                    if target_dataset == 'bank' or target_dataset == 'givemesomecredit':
                        inp = inp.float()
                    a1 = model_bottom[id_attacker](inp[i_id_selected_in_batch].unsqueeze(0).to(device))
                    embedding_library[embedding_count] = a1
                    embedding_count += 1
                if int(label[id_selected_in_batch[i_id_selected_in_batch]].cpu()) == target_label:
                    true_target += 1
            else:
                g_norm[i_id_selected_in_batch, 2] = -1
                
        inp = input[id_attacker]
        inp = inp.to(device)
        if target_dataset == 'bank' or target_dataset == 'givemesomecredit':
            inp = inp.float()
        a1 = model_bottom[id_attacker](inp)
        for i_id_selected_in_batch in range(len(id_selected_in_batch)):
            if id_selected_in_batch[i_id_selected_in_batch]+id*batch_size in train_dataset_correct_list:
                D_id.append(id_selected_in_batch[i_id_selected_in_batch]+id*batch_size)
                D_label = torch.cat((D_label, torch.tensor([int(g_norm[i_id_selected_in_batch, 2])]).detach().to(device)), 0)
                D_embedding = torch.cat((D_embedding, a1[id_selected_in_batch[i_id_selected_in_batch]].unsqueeze(0).detach()), 0)

        id = id + 1

    print(total_target, true_target)
    h_a1.remove()
    h_y.remove()

    # D_label is the presumed labels for the data D_id
    # M_D1_label is the target label data is 1 and others are 0
    # labels[D_id] is the true labels for D_id

    # presumed labels confusion matrix
    print(green("label inference information: ", 'bold'))
    confusion_matrix = np.zeros((labels_kind, labels_kind), int)
    presion_label_inference = 0
    for i in range(len(D_id)):
        confusion_matrix[int(all_labels[D_id[i]]), int(D_label[i])] += 1
        if all_labels[D_id[i]] == D_label[i]:
            presion_label_inference += 1
    presion_label_inference = presion_label_inference/len(D_id)
    print("confusion_matrix: \n", confusion_matrix)
    print("overall_presion_label_inference: ", presion_label_inference)

    print(green("target label information: ", 'bold'))
    M_D1_label = torch.tensor(list(map(lambda x: 1.0 if x == target_label else 0.0, D_label))).to(device)
    print("target label in the D_label", torch.sum(M_D1_label))
    M_efficiency = torch.sum(M_D1_label)/len(M_D1_label)

    num = 0
    for i_D_label in range(len(D_label)):
        if D_label[i_D_label] == target_label and all_labels[D_id[i_D_label]] == target_label:
            num += 1
    train_precision = num/torch.sum(M_D1_label)
    print("train_precision of target label in the D_label", train_precision)

    D_id_target_label_all = []
    for i_D_label in range(len(D_label)):
        if D_label[i_D_label] == target_label:
            D_id_target_label_all.append(D_id[i_D_label])

    print(green("RESULT RECORD: ", 'bold'))
    print("train_precision", train_precision)
    print("M_efficiency", M_efficiency)
    print("target_id_number", torch.sum(M_D1_label))
    print("overall_presion", presion_label_inference)
    print("test_recall", true_target/all_target)

    return D_id, D_label, M_D1, D_id_target_label_all, M_D1_label, embedding_library


# In[176]:


def backdoor_attack_stripe(D_id_target_label_all, batch_size, M_D1_label, all_labels, model_bottom, model_top,
    opt_bottom, opt_top, train_data, labels, val_data, val_labels, poisoning_rate, wave_amp, backdoor_scope, device,
    backdoor_epochs=60, grad_kind=1, mask=None, noise_p=0.5, noise_range=0.05, mask_size=64, mask_random_flag=1, wave_random_flag=1, n_workers=2, embedding_len=64):
    dl = list(zip(*train_data))
    val_dl = list(zip(*val_data))
    grad = torch.zeros(len(all_labels),embedding_len).to(device)
    for epoch in range(backdoor_epochs):
        D_id_target_label = random.sample(D_id_target_label_all, min(int(len(all_labels)*poisoning_rate), int(torch.sum(M_D1_label))))
        print(epoch,' Epoch')
        print('poisoning_rate', min(int(len(all_labels)*poisoning_rate), int(torch.sum(M_D1_label)))/len(all_labels))
        train_correct = 0
        train_total = 0
        id = 0

        for input,label in zip(dl,labels):
            # input data process
            opt_top.zero_grad()
            for i in range(n_workers):
                opt_bottom[i].zero_grad()
            label = label.to(device)
            if target_dataset == 'bank' or target_dataset == 'givemesomecredit':
                label = label.float()
            # model mode
            for i in range(n_workers):
                if i == id_attacker:
                    model_bottom[i].eval()
                else:
                    model_bottom[i].train()
            model_top.train()
            # output of bottom models
            output = []
            for i in range(n_workers):
                inp = input[i]
                inp = inp.to(device)
                if target_dataset == 'bank' or target_dataset == 'givemesomecredit':
                    inp = inp.float()
                if i == id_attacker:
                    a1 = model_bottom[i](inp)
                    for j in range(len(label)):
                        if (j+batch_size*id) in D_id_target_label:
                            # TODO 1 change the judgment conditions by D_id_target_label
                            if j < batch_size - 1:
                                r = random.randint(0, len(label)-1)
                                a1.data[j] = a1.data[r] - grad[r+batch_size*id]

                            # to change 1, the backdoor added
                            if mask_random_flag == 1:
                                mask_random = np.copy(mask)
                                for mask_i in random.sample(range(0, embedding_len), int(embedding_len/2)):
                                    mask_random[mask_i] = 0
                            else:
                                mask_random = mask
                            if wave_random_flag == 1:
                                a1.data[j] = backdoor_stripe(a1.data[j], wave_amp*random.uniform(0.5,1.2), backdoor_scope, mask=mask_random)
                            else:
                                a1.data[j] = backdoor_stripe(a1.data[j], wave_amp, backdoor_scope, mask=mask_random)
                            
                            if epoch > 10:
                                for mask_i in range(embedding_len):
                                    if mask_random[mask_i] == 0:
                                        a1.data[j][mask_i] = a1.data[j][mask_i] - grad[r+batch_size*id][mask_i] * (5 + (min(epoch, 20)-10)/2)
                        else:
                            if grad_kind and random.random() < noise_p:
                                a1.data[j] = a1.data[j] - grad[j+batch_size*id]
                    output.append(a1)
                else:
                    output.append(model_bottom[i](inp))
        
            a = torch.cat(output,1)
            y = model_top(a)

            h = a1.register_hook(save_grad('a1'))

            criterion = nn.NLLLoss()
            loss = criterion(y, label.long())
            
            train_correct += y.max(1)[1].eq(label).sum().item()
            train_total += y.shape[0]
            
            loss.backward()
            opt_top.step()
            for i in range(n_workers):
                if i != id_attacker:
                    opt_bottom[i].step()

            for i in range(len(label)):
                grad[i+batch_size*id] = torch.clamp(grad[i+batch_size*id] + 1e4 * grads['a1'][i], (-1)*noise_range, noise_range)
                
            id = id + 1

        correct = 0
        total = 0
        for model in model_bottom:
            model.eval()
        model_top.eval()
        for input,label in zip(val_dl,val_labels):
            # input data process
            label = label.to(device)
            if target_dataset == 'bank' or target_dataset == 'givemesomecredit':
                label = label.float()

            # output of bottom models
            output = []
            for i in range(n_workers):
                inp = input[i]
                inp = inp.to(device)
                if target_dataset == 'bank' or target_dataset == 'givemesomecredit':
                    inp = inp.float()
                if i == id_attacker:
                    a1 = model_bottom[i](inp)
                    output.append(a1)
                else:
                    output.append(model_bottom[i](inp))
            a = torch.cat(output,1)
            y = model_top(a)
            correct += y.max(1)[1].eq(label).sum().item()
            total += y.shape[0]
        print(f"Train Accuracy: {100*train_correct/train_total:.3f}%")
        print(f"Val Accuracy: {100*correct/total:.3f}%")

        correct = 0
        total = 0
        for model in model_bottom:
            model.eval()
        model_top.eval()
        for input,label in zip(val_dl,val_labels):
            # input data process
            label = label.to(device)
            if target_dataset == 'bank' or target_dataset == 'givemesomecredit':
                label = label.float()

            # output of bottom models
            output = []
            target_label_num = 0
            for i in range(n_workers):
                inp = input[i]
                inp = inp.to(device)
                if target_dataset == 'bank' or target_dataset == 'givemesomecredit':
                    inp = inp.float()
                if i == id_attacker:
                    a1 = model_bottom[i](inp)
                    for j in range(len(label)):
                        a1.data[j] = backdoor_stripe(a1.data[j], wave_amp, backdoor_scope, mask=mask)
                        if label[j] != target_label:
                            label[j] = target_label
                        else:
                            label[j] = -1
                            target_label_num += 1
                    output.append(a1)
                else:
                    output.append(model_bottom[i](inp))
            # top model
            a = torch.cat(output,1)
            y = model_top(a)
            correct += y.max(1)[1].eq(label).sum().item()
            total += y.shape[0] - target_label_num
        print(f"Backdoor Success Rate: {100*correct/total:.3f}%")

    h.remove()
    return model_bottom, model_top

if os.path.exists(Results_path + str(target_label) + "/" +str(n_workers)+"-workers_"+"top_model"+model_type+".pth"):
    embedding_len = get_embedding_len(model_bottom, model_top, train_data, labels, n_workers)
    for i in range(n_workers):
        model_bottom[i].load_state_dict(torch.load(Results_path + str(target_label) + "/" +str(n_workers)+"-workers_"+ "model"+str(i)+model_type+".pth"))
    model_top.load_state_dict(torch.load(Results_path + str(target_label) + "/" +str(n_workers)+"-workers_"+"top_model"+model_type+".pth"))
else:
    model_bottom, model_top, embedding_len = model_training(model_bottom, model_top, opt_bottom, opt_top, train_data, labels, val_data, val_labels, epochs, device, criterion, n_workers=n_workers)
    for i in range(n_workers):
        model = model_bottom[i]
        torch.save(model.state_dict(), Results_path + str(target_label) + "/" +str(n_workers)+"-workers_"+ "model"+str(i)+model_type+".pth")
    torch.save(model_top.state_dict(), Results_path + str(target_label) + "/" +str(n_workers)+"-workers_"+"top_model"+model_type+".pth")

embedding_mean, embedding_std = all_embedding_statistics(model_bottom, model_top, opt_bottom, opt_top, train_data, labels, criterion, batch_size, device, train_num, labels_kind, n_workers=n_workers)
std = []
_std = []
for i in range(len(embedding_std)):
    _std.append((-1)*float(embedding_std[i]))
    std .append(float(embedding_std[i]))
elements_sort = np.argsort(_std)

train_dataset_correct_list, train_dataset_gradient, train_dataset_gradient_comp = correctness_gradient_list(model_bottom, model_top, train_data, labels, criterion, batch_size, device, train_num, n_workers=n_workers, embedding_len=embedding_len)

embedding_id, embedding = embedding_random_choose(target_label, target_label_candidates_number, all_labels, train_dataset_correct_list, train_dataset_gradient, model_bottom, all_imgs, device, n_workers=n_workers)

# calculate the successful rate of direct replacement
direct_replacement_successful_rate(model_bottom, model_top, val_data, val_labels, embedding, target_label, device, n_workers=n_workers)

# label inference to get D_id, D_label, M_D1
D_id, D_label, M_D1, D_id_target_label_all, M_D1_label, embedding_library = label_inference(model_bottom, model_top, embedding_id, embedding, train_data, labels, target_dataset, target_label, labels_kind, search_N, batch_size, device, n_workers=n_workers, embedding_len=embedding_len)

torch.save(M_D1.state_dict(), Results_path + str(target_label) + "/" + "M_D1"+model_type+".pth")

def backdoor_training_group(model_bottom, model_top, poisoning_rate_str, poisoning_rate_list, wave_multiple_list, mask_size_list, grad_kind_list, noise_p_list, noise_range_list, n_workers=2, embedding_len=64):

    for i_poisoning_rate in range(len(poisoning_rate_list)):
        for grad_kind in grad_kind_list:
            for wave_multiple in wave_multiple_list:
                for noise_range in noise_range_list:
                    for noise_p in noise_p_list:
                        for mask_size in mask_size_list:
                            mask = np.zeros_like(std)
                            wave_amp = 0
                            if mask_size > len(mask):
                                mask_size = len(mask)
                                print('Mask size is too big!')
                                print('Now mask size is equal to ', mask_size)
                            for i in range(mask_size):
                                mask[elements_sort[i]] = 1
                                wave_amp += float(std[elements_sort[i]])
                            wave_amp = wave_multiple * wave_amp/mask_size
                            print('wave_amp:', wave_amp)
                            _model_type = model_type + '_poisoningrate_' + str(poisoning_rate_list[i_poisoning_rate]) +'_noisep_'+ str(noise_p) + '_noiserange_' + str(noise_range) +'_masksize_'+ str(mask_size) + '_wavemultiple_' +str(wave_multiple)+'_gradkind_'+str(grad_kind)+'_stripe'
                            print()
                            print()
                            print()
                            print(_model_type)
                            poisoning_rate = poisoning_rate_list[i_poisoning_rate]
                            model_bottom_cp = copy.deepcopy(model_bottom)
                            model_top_cp = copy.deepcopy(model_top)
                            
                            opt_bottom_cp = []
                            criterion = nn.CrossEntropyLoss()
                            if target_dataset == 'imagenette':
                                for i in range(n_workers):
                                    if i == id_attacker:
                                        opt_bottom_cp.append(torch.optim.SGD(params=model_bottom_cp[i].parameters(),lr=0.005))
                                    else:
                                        opt_bottom_cp.append(torch.optim.SGD(params=model_bottom_cp[i].parameters(),lr=0.0005))
                                opt_top_cp = torch.optim.SGD(params=model_top_cp.parameters(),lr=0.01)
                            else:
                                for i in range(n_workers):
                                    if i == id_attacker:
                                        opt_bottom_cp.append(torch.optim.SGD(params=model_bottom_cp[i].parameters(),lr=0.1))
                                    else:
                                        opt_bottom_cp.append(torch.optim.SGD(params=model_bottom_cp[i].parameters(),lr=0.01))
                                opt_top_cp = torch.optim.SGD(params=model_top_cp.parameters(),lr=0.15)

                            model_bottom_cp, model_top_cp = backdoor_attack_stripe(D_id_target_label_all, batch_size, M_D1_label, all_labels, model_bottom_cp, model_top_cp,
                                opt_bottom_cp, opt_top_cp, train_data, labels, val_data, val_labels, poisoning_rate, wave_amp, backdoor_scope, device,
                                backdoor_epochs=backdoor_epochs, grad_kind=grad_kind, mask=mask, noise_p=noise_p, noise_range=noise_range, 
                                mask_size=mask_size, mask_random_flag=1, wave_random_flag=1, n_workers=n_workers, embedding_len=embedding_len)

                            # save the model
                            if grad_kind:
                                for i in range(n_workers):
                                    model_cp = model_bottom_cp[i]
                                    torch.save(model_cp.state_dict(), Results_path +'Poi' + poisoning_rate_str[i_poisoning_rate] + '/'+ str(target_label) + "/" +str(n_workers)+"-workers_"+ "model"+str(i)+_model_type+"_grad_cp.pth")
                                torch.save(model_top_cp.state_dict(), Results_path + 'Poi' + poisoning_rate_str[i_poisoning_rate] + '/'+ str(target_label) + "/" +str(n_workers)+"-workers_"+"top_model"+_model_type+"_grad_cp.pth")
                            else:
                                for i in range(n_workers):
                                    model_cp = model_bottom_cp[i]
                                    torch.save(model_cp.state_dict(), Results_path +'Poi' + poisoning_rate_str[i_poisoning_rate] + '/'+ str(target_label) + "/" +str(n_workers)+"-workers_"+ "model"+str(i)+_model_type+"_cp.pth")
                                torch.save(model_top_cp.state_dict(), Results_path + 'Poi' + poisoning_rate_str[i_poisoning_rate] + '/'+ str(target_label) + "/" +str(n_workers)+"-workers_"+"top_model"+_model_type+"_cp.pth")
                            print()
                            print()
                            print()

poisoning_rate_list = [0.01]
poisoning_rate_str = ['1']
grad_kind_list = [1]
wave_multiple_list = [0.8, 0.4]
noise_p_list = [noise_p]
noise_range_list = [noise_range]
mask_size_list = [mask_size]
backdoor_scope = 0

backdoor_training_group(model_bottom, model_top, poisoning_rate_str, poisoning_rate_list, wave_multiple_list, mask_size_list, grad_kind_list, noise_p_list, noise_range_list, n_workers=n_workers, embedding_len=embedding_len)
