def param():
    description = "split information"    
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset", type=str, default="Cifar10")
    parser.add_argument("--target_label", type=int, default=0)
    parser.add_argument("--target_label_candidates_number", type=int, default=1)
    parser.add_argument("--test_parameter", type=bool, default=False)
    parser.add_argument("--test_noise", type=bool, default=False)
    parser.add_argument("--test_trigger_fabrication", type=bool, default=False)
    parser.add_argument("--test_sample_n", type=bool, default=False)
    parser.add_argument("--test_server_layer", type=bool, default=False)
    parser.add_argument("--test_lr", type=bool, default=False)
    parser.add_argument("--noise_range", type=float, default=0.03)
    parser.add_argument("--noise_p", type=float, default=0.5)
    parser.add_argument("--mask_size", type=int, default=64)
    parser.add_argument("--poisoning_rate", type=float, default=0.01)
    parser.add_argument("--wave_multiple", type=float, default=0.4)
    parser.add_argument("--backdoor_epochs", type=int, default=60)
    parser.add_argument("--upload_method", type=int, default=0)
    args = parser.parse_args()
    return args


import argparse

args = param()

target_dataset = args.dataset
target_label = args.target_label
target_label_candidates_number = args.target_label_candidates_number
test_parameter = args.test_parameter
test_noise = args.test_noise
test_sample_n = args.test_sample_n
test_server_layer = args.test_server_layer
test_trigger_fabrication = args.test_trigger_fabrication
test_lr = args.test_lr
noise_range = args.noise_range
noise_p = args.noise_p
mask_size = args.mask_size
poisoning_rate = args.poisoning_rate
wave_multiple = args.wave_multiple
backdoor_epochs = args.backdoor_epochs
upload_method = args.upload_method
if upload_method == 0:
    model_type = '_normal'
else:
    model_type = '_'+str(upload_method)+'-upload_normal'
print('')
print('')
print('')
print('Test_parameter: ', test_parameter)
print('Test_server_layer: ', test_server_layer)
print('Test_sample_n: ', test_sample_n)
print('Test_trigger_fabrication: ', test_trigger_fabrication)
print('Test_noise: ', test_noise)
print('')
print('')
print('')

from uuid import uuid4

import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms
import random
import copy
from simple_colors import *
from torch.utils.data.sampler import WeightedRandomSampler

from utils.split_model import *
from tool_split_model import *
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
    sample_n = [5, 7, 10, 14, 18, 20]

if target_dataset == "MNIST":
    epochs = 10
    batch_size = 128
    search_N = 14
    labels_kind = 10
    sample_n = [5, 7, 10, 14, 18, 20]

if target_dataset == 'imagenette':
    epochs = 30
    batch_size = 50
    search_N = 10
    labels_kind = 10
    sample_n = [3, 5, 7, 10, 12, 14]

if target_dataset == 'cinic10':
    epochs = 30
    batch_size = 64
    search_N = 8
    labels_kind = 10
    sample_n = [3, 5, 7, 10, 12, 14]

if target_dataset == 'givemesomecredit':
    epochs = 5
    batch_size = 1000
    search_N = 40
    labels_kind = 2
    sample_n = [12, 20, 28, 40, 48, 56]

if target_dataset == 'bank':
    epochs = 10
    batch_size = 100
    search_N = 6
    labels_kind = 2
    sample_n = [2, 3, 6, 8, 10, 12]
    

print(f"Loading {target_dataset}...")
split_train_dataset, split_val_dataset, train_labels, val_labels = dataset.get_train_test_set(
    target_dataset)
train_num = split_train_dataset[0].shape[0]
val_num = split_val_dataset[0].shape[0]

img0 = split_train_dataset[0]
img1 = split_train_dataset[1]
val_img0 = split_val_dataset[0]
val_img1 = split_val_dataset[1]
labels = train_labels
labels_val = val_labels

if target_dataset == 'bank' or target_dataset == 'givemesomecredit':
    img0 = img0.float()
    img1 = img1.float()
    val_img0 = val_img0.float()
    val_img1 = val_img1.float()
    
def shuffle_data(d1, d2, label):
    ll = []
    for i in range(len(d1)):
        ll.append((d1[i], d2[i], label[i]))
    random.shuffle(ll)
    d1, d2, label = zip(*ll)
    return d1, d2, label

img0, img1, labels = shuffle_data(img0, img1, labels)

dl_1 = torch.utils.data.DataLoader(img0, batch_size=batch_size)
dl_1val = torch.utils.data.DataLoader(val_img0, batch_size=batch_size)

dl_2 = torch.utils.data.DataLoader(img1, batch_size=batch_size)
dl_2val = torch.utils.data.DataLoader(val_img1, batch_size=batch_size)

dl_local = torch.utils.data.DataLoader(labels, batch_size=batch_size)
dl_localval = torch.utils.data.DataLoader(labels_val, batch_size=batch_size)

print(f"Finish loading {target_dataset} dataset")

# 保留中间变量函数
grads = {}

def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook

def outlier_detection(l1_norm_list):
    consistency_constant = 1.4826  # if normal distribution
    median = np.median(l1_norm_list)
    mad = consistency_constant * np.median(np.abs(l1_norm_list - median))
    print(f'median: {median}, MAD: {mad}')
    return median, mad

def embedding_statistics(embedding_library):
    print("Before backdoor attack:")
    num = 0
    for i_embedding in embedding_library:
        norm_list = []
        for i_i_embedding in i_embedding:
            norm_list.append(i_i_embedding.sum().item())
        num += 1
        print("Embedding ",num,":")
        outlier_detection(norm_list)
        print("mean:", np.mean(norm_list))
        print("std:", np.std(norm_list))

def all_embedding_statistics(model1, model2, model3, dl_1, dl_2, dl_local, criterion, batch_size, device, train_num, labels_kind):
    iternum = 0
    embedding_list = []
    label_list = []

    for in1, in2, label in zip(dl_1, dl_2, dl_local):
        in1 = in1.to(device)
        in2 = in2.to(device)
        label = label.to(device)
        if target_dataset == 'bank' or target_dataset == 'givemesomecredit':
            in1 = in1.float()
            in2 = in2.float()
            label = label.float()
        model1.eval()
        model2.eval()
        model3.eval()
        a1 = model1(in1)
        a2 = model2(in2)
        if upload_method == 0:
            a = torch.cat((a1, a2), 1)
        elif upload_method == 1:
            a = a1 + a2
        elif upload_method == 2:
            a = a1 + a2
            a = a/2
        elif upload_method == 3:
            a = torch.max(a1, a2)
        elif upload_method == 4:
            a = a1 * a2
        elif upload_method == 5:
            a = torch.min(a1, a2)
        y = model3(a)
        loss = criterion(y, label.long())
        loss.backward()
        embedding_list.append(a1)
        label_list.append(label)
        iternum += 1

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


def all_embedding_statistics_elements(model1, model2, model3, dl_1, dl_2, dl_local, criterion, batch_size, device, train_num, labels_kind):
    iternum = 0
    embedding_list = []
    label_list = []

    for in1, in2, label in zip(dl_1, dl_2, dl_local):
        in1 = in1.to(device)
        in2 = in2.to(device)
        label = label.to(device)
        if target_dataset == 'bank' or target_dataset == 'givemesomecredit':
            in1 = in1.float()
            in2 = in2.float()
            label = label.float()
        model1.eval()
        model2.eval()
        model3.eval()
        a1 = model1(in1)
        a2 = model2(in2)
        if upload_method == 0:
            a = torch.cat((a1, a2), 1)
        elif upload_method == 1:
            a = a1 + a2
        elif upload_method == 2:
            a = a1 + a2
            a = a/2
        elif upload_method == 3:
            a = torch.max(a1, a2)
        elif upload_method == 4:
            a = a1 * a2
        elif upload_method == 5:
            a = torch.min(a1, a2)
        y = model3(a)
        loss = criterion(y, label.long())
        loss.backward()
        embedding_list.append(a1)
        label_list.append(label)
        iternum += 1

    for i in range(labels_kind+1):
        embedding_mean = torch.zeros_like(embedding_list[0][0])
        embedding_std = torch.zeros_like(embedding_list[0][0])
        for num_in_embedding in range(len(embedding_list[0][0])):
            norm_list = []
            if i == 0:
                print("All Embedding:")
                for j in range(len(embedding_list)):
                    for k in range(len(embedding_list[j])):
                        norm_list.append(embedding_list[j][k][num_in_embedding])
            else:
                print("Embedding with label ",i-1,":")
                for j in range(len(embedding_list)):
                    for k in range(len(embedding_list[j])):
                        if label_list[j][k] == i-1:
                            norm_list.append(embedding_list[j][k][num_in_embedding])
            mean_embedding = np.mean(norm_list)
            std_embedding = np.std(norm_list)
            embedding_mean[num_in_embedding] = mean_embedding
            embedding_std[num_in_embedding] = std_embedding
        print("Mean Embedding:")
        print(embedding_mean)
        print("Std Embedding:")
        print(embedding_std)


def backdoor_stripe(tensor, wave_amp, scope=0, mask=None):
    if scope == 0:
        if wave_amp <0.15:
            wave_amp = 0.15
        for i in range(16):
            if int(mask[i*4 + 0]) == 1:
                tensor[i*4 + 0] = tensor[i*4 + 0] + wave_amp
            if int(mask[i*4 + 1]) == 1:
                tensor[i*4 + 1] = tensor[i*4 + 1] + wave_amp
            if int(mask[i*4 + 2]) == 1:
                tensor[i*4 + 2] = tensor[i*4 + 2] - wave_amp
            if int(mask[i*4 + 3]) == 1:
                tensor[i*4 + 3] = tensor[i*4 + 3] - wave_amp
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


def half_backdoor_stripe(tensor, wave_amp):
    for i in range(16):
        tensor[i*4 + 0] = tensor[i*4 + 0] + 0.5 * wave_amp
        tensor[i*4 + 1] = tensor[i*4 + 1] + 0.5 * wave_amp
        tensor[i*4 + 2] = tensor[i*4 + 2] - 0.5 * wave_amp
        tensor[i*4 + 3] = tensor[i*4 + 3] - 0.5 * wave_amp
    return tensor

def backdoor_exchange(tensor):
    tensor2 = tensor.clone()
    for i in range(64):
        tensor[i] = tensor2[63-i]
    return tensor

def backdoor_stripe_thin(tensor, wave_amp):
    for i in range(32):
        tensor[i*2 + 0] = tensor[i*2 + 0] + wave_amp
        tensor[i*2 + 1] = tensor[i*2 + 1] - wave_amp
    return tensor

# model basic train

def model_training(model1, model2, model3, opt1, opt2, local_opt, dl_1, dl_2, dl_local, dl_1val, dl_2val, dl_localval, epochs, device, criterion):
    for epoch in range(epochs):
        train_correct = 0
        train_total = 0
        model1.train()
        model2.train()
        model3.train()
        for in1, in2, label in zip(dl_1, dl_2, dl_local):
            in1 = in1.to(device)
            in2 = in2.to(device)
            label = label.to(device)
            if target_dataset == 'bank' or target_dataset == 'givemesomecredit':
                in1 = in1.float()
                in2 = in2.float()
                label = label.float()
            opt1.zero_grad()
            opt2.zero_grad()
            local_opt.zero_grad()
            a1 = model1(in1)
            a2 = model2(in2)
            if upload_method == 0:
                a = torch.cat((a1, a2), 1)
            elif upload_method == 1:
                a = a1 + a2
            elif upload_method == 2:
                a = a1 + a2
                a = a/2
            elif upload_method == 3:
                a = torch.max(a1, a2)
            elif upload_method == 4:
                a = a1 * a2
            elif upload_method == 5:
                a = torch.min(a1, a2)
            y = model3(a)
            loss = criterion(y, label.long())
            train_correct += y.max(1)[1].eq(label).sum().item()
            train_total += y.shape[0]
            loss.backward()
            local_opt.step()
            opt1.step()
            opt2.step()
        print(loss)
        correct = 0
        total = 0
        model1.eval()
        model2.eval()
        model3.eval()
        for in_val1, in_val2, label_val in zip(dl_1val, dl_2val, dl_localval):
            in_val1 = in_val1.to(device)
            in_val2 = in_val2.to(device)
            label_val = label_val.to(device)
            if target_dataset == 'bank' or target_dataset == 'givemesomecredit':
                in_val1 = in_val1.float()
                in_val2 = in_val2.float()
                label_val = label_val.float()
            a1 = model1(in_val1)
            a2 = model2(in_val2)
            if upload_method == 0:
                a = torch.cat((a1, a2), 1)
            elif upload_method == 1:
                a = a1 + a2
            elif upload_method == 2:
                a = a1 + a2
                a = a/2
            elif upload_method == 3:
                a = torch.max(a1, a2)
            elif upload_method == 4:
                a = a1 * a2
            elif upload_method == 5:
                a = torch.min(a1, a2)
            y = model3(a)
            correct += y.max(1)[1].eq(label_val).sum().item()
            total += y.shape[0]
        print(f"Train Accuracy: {100*train_correct/train_total:.3f}%")
        print(f"Val Accuracy: {100*correct/total:.3f}%")
    return model1, model2, model3

# get correct and gradient list

def correctness_gradient_list(model1, model2, model3, dl_1, dl_2, dl_local, criterion, batch_size, device, train_num):
    iternum = 0
    train_dataset_correct_list = []
    train_dataset_gradient = torch.zeros(train_num)
    train_dataset_gradient_comp = torch.zeros(train_num, 64)

    for in1, in2, label in zip(dl_1, dl_2, dl_local):
        in1 = in1.to(device)
        in2 = in2.to(device)
        label = label.to(device)
        if target_dataset == 'bank' or target_dataset == 'givemesomecredit':
            in1 = in1.float()
            in2 = in2.float()
            label = label.float()
        model1.eval()
        model2.eval()
        model3.eval()
        a1 = model1(in1)
        a2 = model2(in2)
        if upload_method == 0:
            a = torch.cat((a1, a2), 1)
        elif upload_method == 1:
            a = a1 + a2
        elif upload_method == 2:
            a = a1 + a2
            a = a/2
        elif upload_method == 3:
            a = torch.max(a1, a2)
        elif upload_method == 4:
            a = a1 * a2
        elif upload_method == 5:
            a = torch.min(a1, a2)
        y = model3(a)
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

def embedding_random_choose(target_label, target_label_candidates_number, labels, train_dataset_correct_list, train_dataset_gradient, model1, img0, device='cuda'):
    
    target_label_list = []
    target_label_candidates = []

    for i in range(len(labels)):
        if i in train_dataset_correct_list and int(labels[i].item()) == target_label:
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

def direct_replacement_successful_rate(model1, model2, model3, dl_1val, dl_2val, dl_localval, embedding, target_label, device='cuda'):
    correct = 0
    total = 0
    for in_val1, in_val2, label_val in zip(dl_1val, dl_2val, dl_localval):
        in_val1 = in_val1.to(device)
        in_val2 = in_val2.to(device)
        label_val = label_val.to(device)
        if target_dataset == 'bank' or target_dataset == 'givemesomecredit':
            in_val1 = in_val1.float()
            in_val2 = in_val2.float()
            label_val = label_val.float()
        model1.eval()
        model2.eval()
        model3.eval()
        a1 = model1(in_val1)
        a2 = model2(in_val2)
        for i in range(len(label_val)):
            a1.data[i] = embedding[0]
            label_val[i] = target_label
        if upload_method == 0:
            a = torch.cat((a1, a2), 1)
        elif upload_method == 1:
            a = a1 + a2
        elif upload_method == 2:
            a = a1 + a2
            a = a/2
        elif upload_method == 3:
            a = torch.max(a1, a2)
        elif upload_method == 4:
            a = a1 * a2
        elif upload_method == 5:
            a = torch.min(a1, a2)
        y = model3(a)
        correct += y.max(1)[1].eq(label_val).sum().item()
        total += y.shape[0]
    print(f"Backdoor Success Rate: {100*correct/total:.3f}%")

# label inference

def label_inference(model1, model2, model3, embedding_id, embedding, dl_1, dl_2, dl_local, target_dataset, target_label, labels_kind, search_N, batch_size, device, random_samples_flag=0):
    M_D1 = M_Discriminator().to(device)

    # inference
    D_id = [embedding_id]
    D_label = torch.tensor([1]).to(device)
    D_embedding = embedding.squeeze(0).clone()
    D_label = D_label.detach()
    D_embedding = D_embedding.unsqueeze(0).detach()

    # dynamic embedding
    embedding_num = 20
    embedding_count = 1
    embedding_library = torch.Tensor(np.ones([embedding_num, 64])).to(device)
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

    for in1, in2, label in zip(dl_1, dl_2, dl_local):
        in1 = in1.to(device)
        in2 = in2.to(device)
        label = label.to(device)
        if target_dataset == 'bank' or target_dataset == 'givemesomecredit':
            in1 = in1.float()
            in2 = in2.float()
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
                M_in = M_data[:, 0:64]
                M_label = M_data[:, 64].squeeze()
                local_opt_M.zero_grad()
                output = M_D1(M_in).squeeze()
                loss_M = adversarial_loss(output, M_label)
                loss_M.backward()
                local_opt_M.step()

            # select N pieces of data in a batch
            # change the selected function to get the most target label data with model M
            a1 = model1(in1)
            probability = M_D1(a1)
            if target_dataset == 'bank' and in1.shape[0] < batch_size:
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
            model1.eval()
            model2.eval()
            model3.eval()

            a1 = model1(in1)
            a2 = model2(in2)
            if i == 1:
                for j in id_selected_in_batch:
                    a1.data[j] = embedding_library[int(random.random()*embedding_count)]

            if upload_method == 0:
                a = torch.cat((a1, a2), 1)
            elif upload_method == 1:
                a = a1 + a2
            elif upload_method == 2:
                a = a1 + a2
                a = a/2
            elif upload_method == 3:
                a = torch.max(a1, a2)
            elif upload_method == 4:
                a = a1 * a2
            elif upload_method == 5:
                a = torch.min(a1, a2)
            y = model3(a)

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
                    a1 = model1(in1[i_id_selected_in_batch].unsqueeze(0).to(device))
                    embedding_library[embedding_count] = a1
                    embedding_count += 1
                if int(label[id_selected_in_batch[i_id_selected_in_batch]].cpu()) == target_label:
                    true_target += 1
            else:
                g_norm[i_id_selected_in_batch, 2] = -1
                
        a1 = model1(in1)
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
        confusion_matrix[int(labels[D_id[i]]), int(D_label[i])] += 1
        if labels[D_id[i]] == D_label[i]:
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
        if D_label[i_D_label] == target_label and labels[D_id[i_D_label]] == target_label:
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

# backdoor attack

def backdoor_attack_stripe(D_id_target_label_all, batch_size, M_D1_label, labels, model1_cp, model2_cp, model3_cp,
    opt1_cp, opt2_cp, local_opt_cp, dl_1, dl_2, dl_local, dl_1val, dl_2val, dl_localval, poisoning_rate, wave_amp, backdoor_scope, device,
    backdoor_epochs=40, grad_kind=1, mask=None, noise_p=0.5, noise_range=0.05, mask_size=64, mask_random_flag=1, wave_random_flag=1):
    
    grad = torch.zeros(len(labels),64).to(device)
    for epoch in range(backdoor_epochs):
        # poisoning rate 
        D_id_target_label = random.sample(D_id_target_label_all, min(int(len(labels)*poisoning_rate), int(torch.sum(M_D1_label))))
        print(epoch,' Epoch')
        print('poisoning_rate', min(int(len(labels)*poisoning_rate), int(torch.sum(M_D1_label)))/len(labels))
        train_correct = 0
        train_total = 0
        id = 0

        for in1, in2, label in zip(dl_1, dl_2, dl_local):
            in1 = in1.to(device)
            in2 = in2.to(device)
            label = label.to(device)
            if target_dataset == 'bank' or target_dataset == 'givemesomecredit':
                in1 = in1.float()
                in2 = in2.float()
                label = label.float()
            opt1_cp.zero_grad()
            opt2_cp.zero_grad()
            local_opt_cp.zero_grad()
            model1_cp.eval()
            model2_cp.train()
            model3_cp.train()

            a1 = model1_cp(in1)
            a2 = model2_cp(in2)
            
            for i in range(len(label)):
                if (i+batch_size*id) in D_id_target_label:
                    # TODO 1 change the judgment conditions by D_id_target_label
                    if i < batch_size - 1:
                        r = random.randint(0, len(label)-1)
                        a1.data[i] = a1.data[r] - grad[r+batch_size*id]
                    if mask_random_flag == 1:
                        mask_random = np.copy(mask)
                        for mask_i in random.sample(range(0, 64), 32):
                            mask_random[mask_i] = 0
                    else:
                        mask_random = mask
                    if wave_random_flag == 1:
                        a1.data[i] = backdoor_stripe(a1.data[i], wave_amp*random.uniform(0.5,1.2), backdoor_scope, mask=mask_random)
                    else:
                        a1.data[i] = backdoor_stripe(a1.data[i], wave_amp, backdoor_scope, mask=mask_random)
                    
                    if epoch > 10:
                        for mask_i in range(64):
                            if mask_random[mask_i] == 0:
                                a1.data[i][mask_i] = a1.data[i][mask_i] - grad[r+batch_size*id][mask_i] * (5 + (min(epoch, 20)-10)/2)
                else:
                    if grad_kind and random.random() < noise_p:
                        a1.data[i] = a1.data[i] - grad[i+batch_size*id]       
                
            if upload_method == 0:
                a = torch.cat((a1, a2), 1)
            elif upload_method == 1:
                a = a1 + a2
            elif upload_method == 2:
                a = a1 + a2
                a = a/2
            elif upload_method == 3:
                a = torch.max(a1, a2)
            elif upload_method == 4:
                a = a1 * a2
            elif upload_method == 5:
                a = torch.min(a1, a2)
            y = model3_cp(a)

            h = a1.register_hook(save_grad('a1'))

            criterion = nn.NLLLoss()
            loss = criterion(y, label.long())
            
            train_correct += y.max(1)[1].eq(label).sum().item()
            train_total += y.shape[0]
            
            loss.backward()
            local_opt_cp.step()
            opt2_cp.step()

            for i in range(len(label)):
                grad[i+batch_size*id] = torch.clamp(grad[i+batch_size*id] + 1e4 * grads['a1'][i], (-1)*noise_range, noise_range)

            id = id + 1

        correct = 0
        total = 0
        for in_val1, in_val2, label_val in zip(dl_1val, dl_2val, dl_localval):
            in_val1 = in_val1.to(device)
            in_val2 = in_val2.to(device)
            label_val = label_val.to(device)
            if target_dataset == 'bank' or target_dataset == 'givemesomecredit':
                in_val1 = in_val1.float()
                in_val2 = in_val2.float()
                label_val = label_val.float()
            model1_cp.eval()
            model2_cp.eval()
            model3_cp.eval()
            a1 = model1_cp(in_val1)
            a2 = model2_cp(in_val2)
            if upload_method == 0:
                a = torch.cat((a1, a2), 1)
            elif upload_method == 1:
                a = a1 + a2
            elif upload_method == 2:
                a = a1 + a2
                a = a/2
            elif upload_method == 3:
                a = torch.max(a1, a2)
            elif upload_method == 4:
                a = a1 * a2
            elif upload_method == 5:
                a = torch.min(a1, a2)
            y = model3_cp(a)
            correct += y.max(1)[1].eq(label_val).sum().item()
            total += y.shape[0]
        print(f"Train Accuracy: {100*train_correct/train_total:.3f}%")
        print(f"Val Accuracy: {100*correct/total:.3f}%")

        correct = 0
        total = 0
        for in_val1, in_val2, label_val in zip(dl_1val, dl_2val, dl_localval):
            in_val1 = in_val1.to(device)
            in_val2 = in_val2.to(device)
            label_val = label_val.to(device)
            if target_dataset == 'bank' or target_dataset == 'givemesomecredit':
                in_val1 = in_val1.float()
                in_val2 = in_val2.float()
                label_val = label_val.float()
            model1_cp.eval()
            model2_cp.eval()
            model3_cp.eval()
            a1 = model1_cp(in_val1)
            a2 = model2_cp(in_val2)
            target_label_num = 0
            for i in range(len(label_val)):
                a1.data[i] = backdoor_stripe(a1.data[i], wave_amp, backdoor_scope, mask=mask)
                if label_val[i] != target_label:
                    label_val[i] = target_label
                else:
                    label_val[i] = -1
                    target_label_num += 1
            if upload_method == 0:
                a = torch.cat((a1, a2), 1)
            elif upload_method == 1:
                a = a1 + a2
            elif upload_method == 2:
                a = a1 + a2
                a = a/2
            elif upload_method == 3:
                a = torch.max(a1, a2)
            elif upload_method == 4:
                a = a1 * a2
            elif upload_method == 5:
                a = torch.min(a1, a2)
            y = model3_cp(a)
            correct += y.max(1)[1].eq(label_val).sum().item()
            total += y.shape[0] - target_label_num
        print(f"Backdoor Success Rate: {100*correct/total:.3f}%")

    h.remove()
    return model1_cp, model2_cp, model3_cp

model1, model2, model3 = model_construction(target_dataset,upload_method=upload_method)
opt1 = torch.optim.SGD(params=model1.parameters(),lr=0.1)
opt2 = torch.optim.SGD(params=model2.parameters(),lr=0.01)
local_opt = torch.optim.SGD(params=model3.parameters(),lr=0.1)

criterion = nn.CrossEntropyLoss()

if os.path.exists(Results_path + str(target_label) + "/" + "model1"+model_type+".pth"):
    model1.load_state_dict(torch.load(Results_path + str(target_label) + "/" + "model1"+model_type+".pth"))
    model2.load_state_dict(torch.load(Results_path + str(target_label) + "/" + "model2"+model_type+".pth"))
    model3.load_state_dict(torch.load(Results_path + str(target_label) + "/" + "model3"+model_type+".pth"))
else:
    model1, model2, model3 = model_training(model1, model2, model3, opt1, opt2, local_opt, dl_1, dl_2, dl_local, dl_1val, dl_2val, dl_localval, epochs, device, criterion)
    torch.save(model1.state_dict(), Results_path + str(target_label) + "/" + "model1"+model_type+".pth")
    torch.save(model2.state_dict(), Results_path + str(target_label) + "/" + "model2"+model_type+".pth")
    torch.save(model3.state_dict(), Results_path + str(target_label) + "/" + "model3"+model_type+".pth")

embedding_mean, embedding_std = all_embedding_statistics(model1, model2, model3, dl_1, dl_2, dl_local, criterion, batch_size, device, train_num, labels_kind)
std = []
_std = []
for i in range(len(embedding_std)):
    _std.append((-1)*float(embedding_std[i]))
    std .append(float(embedding_std[i]))
elements_sort = np.argsort(_std)

train_dataset_correct_list, train_dataset_gradient, train_dataset_gradient_comp = correctness_gradient_list(model1, model2, model3, dl_1, dl_2, dl_local, criterion, batch_size, device, train_num)

embedding_id, embedding = embedding_random_choose(target_label, target_label_candidates_number, labels, train_dataset_correct_list, train_dataset_gradient, model1, img0, device)

# calculate the successful rate of direct replacement
direct_replacement_successful_rate(model1, model2, model3, dl_1val, dl_2val, dl_localval, embedding, target_label, device)

test_random_samples_flag_list = [0, 1]
if test_sample_n:
    for test_random_samples_flag in test_random_samples_flag_list:
        for i_sample_n in range(len(sample_n)):
            backdoor_scope = 0
            search_N_test = sample_n[i_sample_n]
            print('')
            print('')
            print('')
            print('')
            print('Ramdom Samples:', test_random_samples_flag)
            print("search_N: ", search_N_test)
            D_id, D_label, M_D1, D_id_target_label_all, M_D1_label, embedding_library = label_inference(model1, model2, model3, embedding_id, embedding, dl_1, dl_2, 
            dl_local, target_dataset, target_label, labels_kind, search_N_test, batch_size, device, random_samples_flag=test_random_samples_flag)


            model1_cp = copy.deepcopy(model1)
            model2_cp = copy.deepcopy(model2)
            model3_cp = copy.deepcopy(model3)

            opt1_cp = torch.optim.SGD(params=model1_cp.parameters(),lr=0.1)
            opt2_cp = torch.optim.SGD(params=model2_cp.parameters(),lr=0.01)
            local_opt_cp = torch.optim.SGD(params=model3_cp.parameters(),lr=0.15)
            if target_dataset == 'imagenette':
                opt1_cp = torch.optim.SGD(params=model1_cp.parameters(),lr=0.005)
                opt2_cp = torch.optim.SGD(params=model2_cp.parameters(),lr=0.0005)
                local_opt_cp = torch.optim.SGD(params=model3_cp.parameters(),lr=0.01)

            mask = np.zeros_like(std)
            wave_amp = 0
            for i in range(mask_size):
                mask[elements_sort[i]] = 1
                wave_amp += float(std[elements_sort[i]])
            wave_amp = wave_multiple * wave_amp/mask_size
            print('wave_amp:', wave_amp)

            model1_cp, model2_cp, model3_cp = backdoor_attack_stripe(D_id_target_label_all, batch_size, M_D1_label, labels, model1_cp, model2_cp, model3_cp, 
            opt1_cp, opt2_cp, local_opt_cp, dl_1, dl_2, dl_local, dl_1val, dl_2val, dl_localval, poisoning_rate, wave_amp, backdoor_scope, device, 
            backdoor_epochs=backdoor_epochs, grad_kind=1, mask=mask, noise_p=noise_p, noise_range=noise_range, mask_size=mask_size)

gradient = torch.zeros(64)
gradient_target = torch.zeros(64)
for i in range(len(train_dataset_gradient)):
    gradient = gradient + torch.abs(train_dataset_gradient_comp[i])
    if labels[i] == target_label:
        gradient_target = gradient_target + torch.abs(train_dataset_gradient_comp[i])

print(gradient)
print(gradient_target)

print(torch.topk(gradient, 5))
print(torch.topk(-gradient, 5))
print(torch.topk(gradient_target, 5))
print(torch.topk(-gradient_target, 5))

# label inference to get D_id, D_label, M_D1
D_id, D_label, M_D1, D_id_target_label_all, M_D1_label, embedding_library = label_inference(model1, model2, model3, embedding_id, embedding, 
dl_1, dl_2, dl_local, target_dataset, target_label, labels_kind, search_N, batch_size, device)

torch.save(M_D1.state_dict(), Results_path + str(target_label) + "/" + "M_D1"+model_type+".pth")

if test_trigger_fabrication:
    backdoor_scope = 0
    test_mask_size_list = [64, 32, 16, 8, 4]
    for test_mask_size in test_mask_size_list:
        print('')
        print('')
        print('')
        print('Mask Size = ', test_mask_size)
        print('Mode 0:')
        model1_cp = copy.deepcopy(model1)
        model2_cp = copy.deepcopy(model2)
        model3_cp = copy.deepcopy(model3)

        opt1_cp = torch.optim.SGD(params=model1_cp.parameters(),lr=0.1)
        opt2_cp = torch.optim.SGD(params=model2_cp.parameters(),lr=0.01)
        local_opt_cp = torch.optim.SGD(params=model3_cp.parameters(),lr=0.15)
        if target_dataset == 'imagenette':
            opt1_cp = torch.optim.SGD(params=model1_cp.parameters(),lr=0.005)
            opt2_cp = torch.optim.SGD(params=model2_cp.parameters(),lr=0.0005)
            local_opt_cp = torch.optim.SGD(params=model3_cp.parameters(),lr=0.01)
            
        mask = np.zeros_like(std)
        mask_index = random.sample(range(0, 64), test_mask_size)
        wave_amp = 0
        for i in range(test_mask_size):
            mask[mask_index[i]] = 1
            wave_amp += float(std[mask_index[i]])
        wave_amp = wave_multiple * wave_amp/test_mask_size
        print('wave_amp:', wave_amp)

        model1_cp, model2_cp, model3_cp = backdoor_attack_stripe(D_id_target_label_all, batch_size, M_D1_label, labels, model1_cp, model2_cp, model3_cp, 
            opt1_cp, opt2_cp, local_opt_cp, dl_1, dl_2, dl_local, dl_1val, dl_2val, dl_localval, poisoning_rate, wave_amp, backdoor_scope, device,
            backdoor_epochs=backdoor_epochs, grad_kind=1, mask=mask, noise_p=noise_p, noise_range=noise_range, mask_size=test_mask_size, mask_random_flag=0, wave_random_flag=0)
        
        print('')
        print('')
        print('')
        print('Mask Size = ', test_mask_size)
        print('Mode 1:')
        model1_cp = copy.deepcopy(model1)
        model2_cp = copy.deepcopy(model2)
        model3_cp = copy.deepcopy(model3)

        opt1_cp = torch.optim.SGD(params=model1_cp.parameters(),lr=0.1)
        opt2_cp = torch.optim.SGD(params=model2_cp.parameters(),lr=0.01)
        local_opt_cp = torch.optim.SGD(params=model3_cp.parameters(),lr=0.15)
        if target_dataset == 'imagenette':
            opt1_cp = torch.optim.SGD(params=model1_cp.parameters(),lr=0.005)
            opt2_cp = torch.optim.SGD(params=model2_cp.parameters(),lr=0.0005)
            local_opt_cp = torch.optim.SGD(params=model3_cp.parameters(),lr=0.01)
        
        mask = np.zeros_like(std)
        wave_amp = 0
        for i in range(test_mask_size):
            mask[elements_sort[i]] = 1
            wave_amp += float(std[elements_sort[i]])
        wave_amp = wave_multiple * wave_amp/test_mask_size
        print('wave_amp:', wave_amp)

        model1_cp, model2_cp, model3_cp = backdoor_attack_stripe(D_id_target_label_all, batch_size, M_D1_label, labels, model1_cp, model2_cp, model3_cp, 
        opt1_cp, opt2_cp, local_opt_cp, dl_1, dl_2, dl_local, dl_1val, dl_2val, dl_localval, poisoning_rate, wave_amp, backdoor_scope, device,
        backdoor_epochs=backdoor_epochs, grad_kind=1, mask=mask, noise_p=noise_p, noise_range=noise_range, mask_size=test_mask_size, mask_random_flag=0, wave_random_flag=0)
        
        print('')
        print('')
        print('')
        print('Mask Size = ', test_mask_size)
        print('Mode 1+2:')
        model1_cp = copy.deepcopy(model1)
        model2_cp = copy.deepcopy(model2)
        model3_cp = copy.deepcopy(model3)

        opt1_cp = torch.optim.SGD(params=model1_cp.parameters(),lr=0.1)
        opt2_cp = torch.optim.SGD(params=model2_cp.parameters(),lr=0.01)
        local_opt_cp = torch.optim.SGD(params=model3_cp.parameters(),lr=0.15)
        if target_dataset == 'imagenette':
            opt1_cp = torch.optim.SGD(params=model1_cp.parameters(),lr=0.005)
            opt2_cp = torch.optim.SGD(params=model2_cp.parameters(),lr=0.0005)
            local_opt_cp = torch.optim.SGD(params=model3_cp.parameters(),lr=0.01)
        
        mask = np.zeros_like(std)
        wave_amp = 0
        for i in range(test_mask_size):
            mask[elements_sort[i]] = 1
            wave_amp += float(std[elements_sort[i]])
        wave_amp = wave_multiple * wave_amp/test_mask_size
        print('wave_amp:', wave_amp)

        model1_cp, model2_cp, model3_cp = backdoor_attack_stripe(D_id_target_label_all, batch_size, M_D1_label, labels, model1_cp, model2_cp, model3_cp, 
        opt1_cp, opt2_cp, local_opt_cp, dl_1, dl_2, dl_local, dl_1val, dl_2val, dl_localval, poisoning_rate, wave_amp, backdoor_scope, device,
        backdoor_epochs=backdoor_epochs, grad_kind=1, mask=mask, noise_p=noise_p, noise_range=noise_range, mask_size=test_mask_size, mask_random_flag=0)
        
        print('')
        print('')
        print('')
        print('Mode 1+2+3:')
        model1_cp = copy.deepcopy(model1)
        model2_cp = copy.deepcopy(model2)
        model3_cp = copy.deepcopy(model3)

        opt1_cp = torch.optim.SGD(params=model1_cp.parameters(),lr=0.1)
        opt2_cp = torch.optim.SGD(params=model2_cp.parameters(),lr=0.01)
        local_opt_cp = torch.optim.SGD(params=model3_cp.parameters(),lr=0.15)
        if target_dataset == 'imagenette':
            opt1_cp = torch.optim.SGD(params=model1_cp.parameters(),lr=0.005)
            opt2_cp = torch.optim.SGD(params=model2_cp.parameters(),lr=0.0005)
            local_opt_cp = torch.optim.SGD(params=model3_cp.parameters(),lr=0.01)
        
        mask = np.zeros_like(std)
        wave_amp = 0
        for i in range(test_mask_size):
            mask[elements_sort[i]] = 1
            wave_amp += float(std[elements_sort[i]])
        wave_amp = wave_multiple * wave_amp/test_mask_size
        print('wave_amp:', wave_amp)

        model1_cp, model2_cp, model3_cp = backdoor_attack_stripe(D_id_target_label_all, batch_size, M_D1_label, labels, model1_cp, model2_cp, model3_cp, 
        opt1_cp, opt2_cp, local_opt_cp, dl_1, dl_2, dl_local, dl_1val, dl_2val, dl_localval, poisoning_rate, wave_amp, backdoor_scope, device, 
        backdoor_epochs=backdoor_epochs, grad_kind=1, mask=mask, noise_p=noise_p, noise_range=noise_range, mask_size=test_mask_size)


if test_noise:
    backdoor_scope = 0
    test_noise_range_list = [0, 0.01, 0.03,0.05, 0.07, 0.09]
    test_mask_size_list = [64, 32, 16, 8]

    for test_mask_size in test_mask_size_list:
        for test_noise_range in test_noise_range_list:
            print('')
            print('')
            print('')
            print('Mask Size:', test_mask_size)
            print('Noise Range:', test_noise_range)
            model1_cp = copy.deepcopy(model1)
            model2_cp = copy.deepcopy(model2)
            model3_cp = copy.deepcopy(model3)

            opt1_cp = torch.optim.SGD(params=model1_cp.parameters(),lr=0.1)
            opt2_cp = torch.optim.SGD(params=model2_cp.parameters(),lr=0.01)
            local_opt_cp = torch.optim.SGD(params=model3_cp.parameters(),lr=0.15)
            if target_dataset == 'imagenette':
                opt1_cp = torch.optim.SGD(params=model1_cp.parameters(),lr=0.005)
                opt2_cp = torch.optim.SGD(params=model2_cp.parameters(),lr=0.0005)
                local_opt_cp = torch.optim.SGD(params=model3_cp.parameters(),lr=0.01)

            mask = np.zeros_like(std)
            wave_amp = 0
            for i in range(test_mask_size):
                mask[elements_sort[i]] = 1
                wave_amp += float(std[elements_sort[i]])
            wave_amp = wave_multiple * wave_amp/test_mask_size
            print('wave_amp:', wave_amp)

            if test_noise_range == 0:
                model1_cp, model2_cp, model3_cp = backdoor_attack_stripe(D_id_target_label_all, batch_size, M_D1_label, labels, model1_cp, model2_cp, model3_cp, 
                    opt1_cp, opt2_cp, local_opt_cp, dl_1, dl_2, dl_local, dl_1val, dl_2val, dl_localval, poisoning_rate, wave_amp, backdoor_scope, device,
                    backdoor_epochs=backdoor_epochs, grad_kind=0, mask=mask, noise_p=noise_p, noise_range=test_noise_range, mask_size=test_mask_size, mask_random_flag=1)
            else:
                model1_cp, model2_cp, model3_cp = backdoor_attack_stripe(D_id_target_label_all, batch_size, M_D1_label, labels, model1_cp, model2_cp, model3_cp, 
                    opt1_cp, opt2_cp, local_opt_cp, dl_1, dl_2, dl_local, dl_1val, dl_2val, dl_localval, poisoning_rate, wave_amp, backdoor_scope, device,
                    backdoor_epochs=backdoor_epochs, grad_kind=1, mask=mask, noise_p=noise_p, noise_range=test_noise_range, mask_size=test_mask_size, mask_random_flag=1)

def backdoor_training_group(poisoning_rate_str, poisoning_rate_list, wave_multiple_list, mask_size_list, grad_kind_list, noise_p_list, noise_range_list):

    for i_poisoning_rate in range(len(poisoning_rate_list)):
        for grad_kind in grad_kind_list:
            for wave_multiple in wave_multiple_list:
                for noise_range in noise_range_list:
                    for noise_p in noise_p_list:
                        for mask_size in mask_size_list:
                            mask = np.zeros_like(std)
                            wave_amp = 0
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
                            model1_cp = copy.deepcopy(model1)
                            model2_cp = copy.deepcopy(model2)
                            model3_cp = copy.deepcopy(model3)

                            opt1_cp = torch.optim.SGD(params=model1_cp.parameters(),lr=0.1)
                            opt2_cp = torch.optim.SGD(params=model2_cp.parameters(),lr=0.01)
                            local_opt_cp = torch.optim.SGD(params=model3_cp.parameters(),lr=0.15)
                            if target_dataset == 'imagenette':
                                opt1_cp = torch.optim.SGD(params=model1_cp.parameters(),lr=0.005)
                                opt2_cp = torch.optim.SGD(params=model2_cp.parameters(),lr=0.0005)
                                local_opt_cp = torch.optim.SGD(params=model3_cp.parameters(),lr=0.01)

                            model1_cp, model2_cp, model3_cp = backdoor_attack_stripe(D_id_target_label_all, batch_size, M_D1_label, labels, model1_cp, model2_cp, model3_cp, 
                            opt1_cp, opt2_cp, local_opt_cp, dl_1, dl_2, dl_local, dl_1val, dl_2val, dl_localval, poisoning_rate, wave_amp, backdoor_scope, device, 
                            backdoor_epochs=backdoor_epochs, grad_kind=grad_kind, mask=mask, noise_p=noise_p, noise_range=noise_range, mask_size=mask_size)

                            # save the model
                            if grad_kind:
                                torch.save(model1_cp.state_dict(), Results_path + 'Poi' + poisoning_rate_str[i_poisoning_rate] + '/' + str(target_label) + "/" + "model1"+_model_type+"_grad_cp.pth")
                                torch.save(model2_cp.state_dict(), Results_path + 'Poi' + poisoning_rate_str[i_poisoning_rate] + '/' + str(target_label) + "/" + "model2"+_model_type+"_grad_cp.pth")
                                torch.save(model3_cp.state_dict(), Results_path + 'Poi' + poisoning_rate_str[i_poisoning_rate] + '/' + str(target_label) + "/" + "model3"+_model_type+"_grad_cp.pth")
                            else:
                                torch.save(model1_cp.state_dict(), Results_path + 'Poi' + poisoning_rate_str[i_poisoning_rate] + '/' + str(target_label) + "/" + "model1"+_model_type+"_cp.pth")
                                torch.save(model2_cp.state_dict(), Results_path + 'Poi' + poisoning_rate_str[i_poisoning_rate] + '/' + str(target_label) + "/" + "model2"+_model_type+"_cp.pth")
                                torch.save(model3_cp.state_dict(), Results_path + 'Poi' + poisoning_rate_str[i_poisoning_rate] + '/' + str(target_label) + "/" + "model3"+_model_type+"_cp.pth")
                            print()
                            print()
                            print()

poisoning_rate_list = [poisoning_rate]
poisoning_rate_str = ['1']
grad_kind_list = [1]
wave_multiple_list = [wave_multiple]
noise_p_list = [noise_p]
noise_range_list = [noise_range]
mask_size_list = [mask_size]
backdoor_scope = 0

if test_parameter == True:
    _poisoning_rate_str = ['1', '1', '1', '1', '1']
    _poisoning_rate_list = [0.005, 0.01, 0.02, 0.03, 0.05]
    _wave_multiple_list = [0.2, 0.4, 0.6, 0.8, 1.0]
    _mask_size_list = [64, 32, 16, 8, 4]

    backdoor_training_group(_poisoning_rate_str, _poisoning_rate_list, wave_multiple_list, mask_size_list, grad_kind_list, noise_p_list, noise_range_list)
    backdoor_training_group(poisoning_rate_str, poisoning_rate_list, _wave_multiple_list, mask_size_list, grad_kind_list, noise_p_list, noise_range_list)
    backdoor_training_group(poisoning_rate_str, poisoning_rate_list, wave_multiple_list, _mask_size_list, grad_kind_list, noise_p_list, noise_range_list)

else:
    backdoor_training_group(poisoning_rate_str, poisoning_rate_list, wave_multiple_list, mask_size_list, grad_kind_list, noise_p_list, noise_range_list)
    

# experiments on learning_rate
if test_lr:
    learning_rate_list = [0.2, 0.15, 0.1, 0.08, 0.05, 0.03, 0.01]
    
    for i_learning_rate in range(len(learning_rate_list)):
        learning_rate = learning_rate_list[i_learning_rate]
        print('')
        print('')
        print('')
        print("learning_rate: ", learning_rate)
        poisoning_rate = 0.01
        backdoor_scope = 0
        model1, model2, model3 = model_construction(target_dataset)
        opt1 = torch.optim.SGD(params=model1.parameters(),lr=learning_rate)
        opt2 = torch.optim.SGD(params=model2.parameters(),lr=0.01)
        local_opt = torch.optim.SGD(params=model3.parameters(),lr=0.1)
        
        criterion = nn.CrossEntropyLoss()

        # train the split model
        model1, model2, model3 = model_training(model1, model2, model3, opt1, opt2, local_opt, 
        dl_1, dl_2, dl_local, dl_1val, dl_2val, dl_localval, epochs, device, criterion)
        
        embedding_mean, embedding_std = all_embedding_statistics(model1, model2, model3, dl_1, dl_2, dl_local, criterion, batch_size, device, train_num, labels_kind)
        std = []
        _std = []
        for i in range(len(embedding_std)):
            _std.append((-1)*float(embedding_std[i]))
            std .append(float(embedding_std[i]))
        elements_sort = np.argsort(_std)
        
        train_dataset_correct_list, train_dataset_gradient, train_dataset_gradient_comp = correctness_gradient_list(model1, model2, model3, dl_1, dl_2, dl_local, criterion, batch_size, device, train_num)

        embedding_id, embedding = embedding_random_choose(target_label, target_label_candidates_number, labels, train_dataset_correct_list, train_dataset_gradient, model1, img0, device)


        direct_replacement_successful_rate(model1, model2, model3, dl_1val, dl_2val, dl_localval, embedding, target_label, device)
        
        # label inference to get D_id, D_label, M_D1
        D_id, D_label, M_D1, D_id_target_label_all, M_D1_label, embedding_library = label_inference(model1, model2, model3, 
            embedding_id, embedding, dl_1, dl_2, dl_local, target_dataset, target_label, labels_kind, search_N, batch_size, device)


        model1_cp = copy.deepcopy(model1)
        model2_cp = copy.deepcopy(model2)
        model3_cp = copy.deepcopy(model3)

        opt1_cp = torch.optim.SGD(params=model1_cp.parameters(),lr=learning_rate)
        opt2_cp = torch.optim.SGD(params=model2_cp.parameters(),lr=0.01)
        local_opt_cp = torch.optim.SGD(params=model3_cp.parameters(),lr=0.1)

        if target_dataset == 'imagenette':
            opt1_cp = torch.optim.Adam(params=model1_cp.parameters(), lr=0.0005*learning_rate/0.01)
            opt2_cp = torch.optim.Adam(params=model2_cp.parameters(), lr=0.0005)
            local_opt_cp = torch.optim.Adam(params=model3_cp.parameters(), lr=0.01)
        
        mask = np.zeros_like(std)
        wave_amp = 0
        for i in range(mask_size):
            mask[elements_sort[i]] = 1
            wave_amp += float(std[elements_sort[i]])
        wave_amp = wave_multiple * wave_amp/mask_size
        print('wave_amp:', wave_amp)

        model1_cp, model2_cp, model3_cp = backdoor_attack_stripe(D_id_target_label_all, batch_size, M_D1_label, labels, model1_cp, model2_cp, model3_cp, 
            opt1_cp, opt2_cp, local_opt_cp, dl_1, dl_2, dl_local, dl_1val, dl_2val, dl_localval, poisoning_rate, wave_amp, backdoor_scope, device, 
            backdoor_epochs=backdoor_epochs, grad_kind=1, mask=mask, noise_p=noise_p, noise_range=noise_range, mask_size=mask_size)


# experiments on learning_rate
if test_server_layer:    
    for server_layer in [1,2,3,4,5]:
        print('')
        print('')
        print('')
        print('server_layer', server_layer)
        backdoor_scope = 0
        model1, model2, model3 = model_construction(target_dataset, server_layer)
        opt1 = torch.optim.SGD(params=model1.parameters(),lr=0.1)
        opt2 = torch.optim.SGD(params=model2.parameters(),lr=0.01)
        local_opt = torch.optim.SGD(params=model3.parameters(),lr=0.1)
        
        criterion = nn.CrossEntropyLoss()

        # train the split model
        model1, model2, model3 = model_training(model1, model2, model3, opt1, opt2, local_opt, 
        dl_1, dl_2, dl_local, dl_1val, dl_2val, dl_localval, epochs, device, criterion)
        
        embedding_mean, embedding_std = all_embedding_statistics(model1, model2, model3, dl_1, dl_2, dl_local, criterion, batch_size, device, train_num, labels_kind)
        std = []
        _std = []
        for i in range(len(embedding_std)):
            _std.append((-1)*float(embedding_std[i]))
            std .append(float(embedding_std[i]))
        elements_sort = np.argsort(_std)
                
        train_dataset_correct_list, train_dataset_gradient, train_dataset_gradient_comp = correctness_gradient_list(model1, model2, model3, dl_1, dl_2, dl_local, criterion, batch_size, device, train_num)

        embedding_id, embedding = embedding_random_choose(target_label, target_label_candidates_number, labels, train_dataset_correct_list, train_dataset_gradient, model1, img0, device)

        direct_replacement_successful_rate(model1, model2, model3, dl_1val, dl_2val, dl_localval, embedding, target_label, device)
        
        # label inference to get D_id, D_label, M_D1
        D_id, D_label, M_D1, D_id_target_label_all, M_D1_label, embedding_library = label_inference(model1, model2, model3, 
            embedding_id, embedding, dl_1, dl_2, dl_local, target_dataset, target_label, labels_kind, search_N, batch_size, device)


        model1_cp = copy.deepcopy(model1)
        model2_cp = copy.deepcopy(model2)
        model3_cp = copy.deepcopy(model3)

        opt1_cp = torch.optim.SGD(params=model1_cp.parameters(),lr=0.1)
        opt2_cp = torch.optim.SGD(params=model2_cp.parameters(),lr=0.01)
        local_opt_cp = torch.optim.SGD(params=model3_cp.parameters(),lr=0.15)
        if target_dataset == 'imagenette':
            opt1_cp = torch.optim.SGD(params=model1_cp.parameters(),lr=0.005)
            opt2_cp = torch.optim.SGD(params=model2_cp.parameters(),lr=0.0005)
            local_opt_cp = torch.optim.SGD(params=model3_cp.parameters(),lr=0.01)

        mask = np.zeros_like(std)
        wave_amp = 0
        for i in range(mask_size):
            mask[elements_sort[i]] = 1
            wave_amp += float(std[elements_sort[i]])
        wave_amp = wave_multiple * wave_amp/mask_size
        print('wave_amp:', wave_amp)

        model1_cp, model2_cp, model3_cp = backdoor_attack_stripe(D_id_target_label_all, batch_size, M_D1_label, labels, model1_cp, model2_cp, model3_cp, 
            opt1_cp, opt2_cp, local_opt_cp, dl_1, dl_2, dl_local, dl_1val, dl_2val, dl_localval, poisoning_rate, wave_amp, backdoor_scope, device, 
            backdoor_epochs=backdoor_epochs, grad_kind=1, mask=mask, noise_p=noise_p, noise_range=noise_range, mask_size=mask_size)

