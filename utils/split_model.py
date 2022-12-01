# In order to serialise our model we need to define it as below
from turtle import forward
from torch import nn
import torch
import math
import torch.nn.functional as F
from torch.nn import init
from torchvision import transforms, datasets, models


hidden_sizes = [128, 500]

output_size = 10


def conv3x3(in_features, out_features):
    return nn.Conv2d(in_features, out_features, kernel_size=3, padding=1)


class SyNet_client(nn.Module):
    def __init__(self):
        super(SyNet_client, self).__init__()
        self.lin = nn.Linear(392, 64)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.lin(x)
        x = nn.functional.relu(x)
        return x


class SyNet_client_four_layers(nn.Module):
    def __init__(self):
        super(SyNet_client_four_layers, self).__init__()
        self.lin1 = nn.Linear(392, 1024)
        self.lin2 = nn.Linear(1024, 512)
        self.lin3 = nn.Linear(512, 128)
        self.lin4 = nn.Linear(128, 64)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.lin1(x)
        x = nn.functional.relu(x)
        x = self.lin2(x)
        x = nn.functional.relu(x)
        x = self.lin3(x)
        x = nn.functional.relu(x)
        x = self.lin4(x)
        x = nn.functional.relu(x)
        return x



class SyNet_server(nn.Module):
    def __init__(self):
        super(SyNet_server, self).__init__()
        self.lin2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.lin3 = nn.Linear(hidden_sizes[1], output_size)
        self.sft = nn.LogSoftmax(dim=1)
        self._sft = nn.Softmax(dim=1)

    def forward(self, x, not_log=False):
        x = self.lin2(x)
        x = nn.ReLU()(x)
        x = self.lin3(x)
        if not_log:
            x = self._sft(x)
        else:
            x = self.sft(x)
        return x


class SyNet_client_cifar10(nn.Module):
    def __init__(self):
        super(SyNet_client_cifar10, self).__init__()
        self.features = nn.Sequential(
            # 1
            conv3x3(3, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 2
            conv3x3(64, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 3
            conv3x3(64, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 4
            conv3x3(128, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 5
            conv3x3(128, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 6
            conv3x3(256, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 7
            conv3x3(256, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 8
            conv3x3(256, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 9
            conv3x3(256, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 10
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 11
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 12
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 13
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 14
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 15
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 16
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            # 17
            nn.Linear(1024, 4096),
            nn.ReLU(),
            nn.Dropout(),
            # 18
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            # 19
            nn.Linear(4096, 64),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

class SyNet_server_cifar10(nn.Module):
    def __init__(self, upload_method=0):
        super(SyNet_server_cifar10, self).__init__()
        if upload_method == 0:
            self.fc = torch.nn.Linear(128, 10)
        elif upload_method > 0:
            self.fc = torch.nn.Linear(64, 10)
        self.sft = nn.LogSoftmax(dim=1)
        self._sft = nn.Softmax(dim=1)

    def forward(self, x, not_log=False):
        x = self.fc(x)
        if not_log:
            x = self._sft(x)
        else:
            x = self.sft(x)
        return x


class SyNet_server_cifar10_2(nn.Module):
    def __init__(self):
        super(SyNet_server_cifar10_2, self).__init__()
        self.fc1 = torch.nn.Linear(128, 128)
        self.fc2 = torch.nn.Linear(128, 10)
        self.sft = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = self.sft(x)
        return x


class SyNet_server_cifar10_3(nn.Module):
    def __init__(self):
        super(SyNet_server_cifar10_3, self).__init__()
        self.fc1 = torch.nn.Linear(128, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 10)
        self.sft = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.fc3(x)
        x = self.sft(x)
        return x


class SyNet_server_cifar10_4(nn.Module):
    def __init__(self):
        super(SyNet_server_cifar10_4, self).__init__()
        self.fc1 = torch.nn.Linear(128, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 32)
        self.fc4 = torch.nn.Linear(32, 10)
        self.sft = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.fc3(x)
        x = nn.ReLU()(x)
        x = self.fc4(x)
        x = self.sft(x)
        return x


class SyNet_server_cifar10_5(nn.Module):
    def __init__(self):
        super(SyNet_server_cifar10_5, self).__init__()
        self.fc1 = torch.nn.Linear(128, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 32)
        self.fc4 = torch.nn.Linear(32, 16)
        self.fc5 = torch.nn.Linear(16, 10)
        self.sft = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.fc3(x)
        x = nn.ReLU()(x)
        x = self.fc4(x)
        x = nn.ReLU()(x)
        x = self.fc5(x)
        x = self.sft(x)
        return x


class SyNet_client_mnist(nn.Module):
    def __init__(self):
        super(SyNet_client_mnist, self).__init__()
        self.lin1 = nn.Linear(512, 1024)
        self.lin2 = nn.Linear(1024, 512)
        self.lin3 = nn.Linear(512, 128)
        self.lin4 = nn.Linear(128, 64)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.lin1(x)
        x = nn.functional.relu(x)
        x = self.lin2(x)
        x = nn.functional.relu(x)
        x = self.lin3(x)
        x = nn.functional.relu(x)
        x = self.lin4(x)
        x = nn.functional.relu(x)
        return x


class SyNet_server_mnist(nn.Module):
    def __init__(self, upload_method=0):
        super(SyNet_server_mnist, self).__init__()
        if upload_method == 0:
            self.lin2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        elif upload_method > 0:
            self.lin2 = nn.Linear(int(hidden_sizes[0]/2), hidden_sizes[1])
        self.lin3 = nn.Linear(hidden_sizes[1], output_size)
        self.sft = nn.LogSoftmax(dim=1)
        self._sft = nn.Softmax(dim=1)

    def forward(self, x, not_log=False):
        x = self.lin2(x)
        x = nn.ReLU()(x)
        x = self.lin3(x)
        if not_log:
            x = self._sft(x)
        else:
            x = self.sft(x)
        return x

class M_Discriminator(nn.Module):
    def __init__(self):
        super(M_Discriminator, self).__init__()
        self.lin1 = nn.Linear(64, 128)
        self.lin3 = nn.Linear(128, 32)
        self.lin4 = nn.Linear(32, 16)
        self.lin2 = nn.Sequential(nn.Linear(16, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin3(x)
        x = F.relu(x)
        x = self.lin4(x)
        x = F.relu(x)
        x = self.lin2(x)
        return x

def CBR(in_channels, out_channels):

    cbr = nn.Sequential(

        nn.Conv2d(in_channels, out_channels,
                  kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
    return cbr


class VGG(nn.Module):

    def __init__(self, block_nums):

        super(VGG, self).__init__()

        self.block1 = self._make_layers(
            in_channels=3, out_channels=64, block_num=block_nums[0])
        self.block2 = self._make_layers(
            in_channels=64, out_channels=128, block_num=block_nums[1])
        self.block3 = self._make_layers(
            in_channels=128, out_channels=256, block_num=block_nums[2])
        self.block4 = self._make_layers(
            in_channels=256, out_channels=512, block_num=block_nums[3])
        self.block5 = self._make_layers(
            in_channels=512, out_channels=512, block_num=block_nums[4])
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(

            nn.Linear(512*7*3, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(4096, 64)
        )

    def _make_layers(self, in_channels, out_channels, block_num):

        blocks = []
        blocks.append(CBR(in_channels, out_channels))

        for i in range(1, block_num):

            blocks.append(CBR(out_channels, out_channels))

        blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))

        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.dropout(x)
        x = self.block3(x)
        x = self.dropout(x)
        x = self.block4(x)
        x = self.block5(x)
        x = torch.flatten(x, start_dim=1)
        out = self.classifier(x)

        return out


def SyNet_client_imagenette():
    block_nums = [2, 2, 3, 3, 3]
    model = VGG(block_nums)
    return model


class SyNet_server_imagenette(nn.Module):
    def __init__(self, upload_method=0):
        super(SyNet_server_imagenette, self).__init__()
        if upload_method == 0:
            self.fc = torch.nn.Linear(128, 10)
        elif upload_method > 0:
            self.fc = torch.nn.Linear(64, 10)
        self.sft = nn.LogSoftmax(dim=1)
        self._sft = nn.Softmax(dim=1)

    def forward(self, x, not_log=False):
        x = self.fc(x)
        if not_log:
            x = self._sft(x)
        else:
            x = self.sft(x)
        return x


D_ = 2 ** 13

def weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class SyNet_server_criteo(nn.Module):

    def __init__(self, upload_method=0):
        super(SyNet_server_criteo, self).__init__()
        if upload_method == 0:
            self.fc1_top = torch.nn.Linear(128, 32)
        elif upload_method > 0:
            self.fc1_top = torch.nn.Linear(64, 32)
        self.fc2_top = nn.Linear(32, 8)
        self.fc3_top = nn.Linear(8, 2)
        self.apply(weights_init)

    def forward(self, x):
        x = F.relu(x)
        x = self.fc1_top(x)
        x = F.relu(x)
        x = self.fc2_top(x)
        x = F.relu(x)
        x = self.fc3_top(x)
        x = F.softmax(x)
        return x


class SyNet_client_criteo(nn.Module):

    def __init__(self):
        super(SyNet_client_criteo, self).__init__()
        self.fc1 = nn.Linear(4096, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(256, 64)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        return x


cfg = {'VGG16':[64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512]}

class SyNet_client_cinic10(nn.Module):
    def _make_layers(self,cfg,in_channels=3):
        layers=[]
        for v in cfg:
            if v=='M':
                layers+=[nn.MaxPool2d(kernel_size=2,stride=2)]
            else:
                layers+=[nn.Conv2d(in_channels,v,kernel_size=3,padding=1),
                         nn.BatchNorm2d(v),
                         nn.ReLU(inplace=True)]
                
                in_channels=v
                
        return nn.Sequential(*layers)
    
    def __init__(self,net_name="VGG16"):
        super(SyNet_client_cinic10,self).__init__()
        
        self.features=self._make_layers(cfg[net_name])
        
        self.classifier=nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024,512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512,512),
            nn.ReLU(True),
            nn.Linear(512,64)
            )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n=m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0,math.sqrt(2./n))
                m.bias.data.zero_()
                
    def forward(self,x):
        x=self.features(x)
        x=x.view(x.size(0),-1)
        x=self.classifier(x)
        return x

class SyNet_server_cinic10(nn.Module):
    def __init__(self, upload_method=0):
        super(SyNet_server_cinic10, self).__init__()
        if upload_method == 0:
            self.fc = torch.nn.Linear(128, 10)
        elif upload_method > 0:
            self.fc = torch.nn.Linear(64, 10)
        self.sft = nn.LogSoftmax(dim=1)
        self._sft = nn.Softmax(dim=1)

    def forward(self, x, not_log=False):
        x = self.fc(x)
        if not_log:
            x = self._sft(x)
        else:
            x = self.sft(x)
        return x

class SyNet_client_credit(nn.Module):

    def __init__(self):
        super(SyNet_client_credit, self).__init__()
        self.fc1 = nn.Linear(5, 32)
        self.fc2 = nn.Linear(32, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 64)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        return x



class SyNet_server_credit(nn.Module):
    def __init__(self, upload_method=0):
        super(SyNet_server_credit, self).__init__()
        if upload_method == 0:
            self.fc = torch.nn.Linear(128, 2)
        elif upload_method > 0:
            self.fc = torch.nn.Linear(64, 2)
        self.sft = nn.LogSoftmax(dim=1)
        self._sft = nn.Softmax(dim=1)

    def forward(self, x, not_log=False):
        x = self.fc(x)
        if not_log:
            x = self._sft(x)
        else:
            x = self.sft(x)
        return x


class SyNet_client_bank(nn.Module):

    def __init__(self):
        super(SyNet_client_bank, self).__init__()
        self.fc1 = nn.Linear(24, 32)
        self.fc2 = nn.Linear(32, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 64)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        return x



class SyNet_server_bank(nn.Module):
    def __init__(self, upload_method=0):
        super(SyNet_server_bank, self).__init__()
        if upload_method == 0:
            self.fc = torch.nn.Linear(128, 2)
        elif upload_method > 0:
            self.fc = torch.nn.Linear(64, 2)
        self.sft = nn.LogSoftmax(dim=1)
        self._sft = nn.Softmax(dim=1)

    def forward(self, x, not_log=False):
        x = self.fc(x)
        if not_log:
            x = self._sft(x)
        else:
            x = self.sft(x)
        return x


class SyNet_server_bank_2(nn.Module):
    def __init__(self):
        super(SyNet_server_bank_2, self).__init__()
        self.fc1 = torch.nn.Linear(128, 128)
        self.fc2 = torch.nn.Linear(128, 2)
        self.sft = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = self.sft(x)
        return x


class SyNet_server_bank_3(nn.Module):
    def __init__(self):
        super(SyNet_server_bank_3, self).__init__()
        self.fc1 = torch.nn.Linear(128, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 2)
        self.sft = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.fc3(x)
        x = self.sft(x)
        return x


class SyNet_server_bank_4(nn.Module):
    def __init__(self):
        super(SyNet_server_bank_4, self).__init__()
        self.fc1 = torch.nn.Linear(128, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 32)
        self.fc4 = torch.nn.Linear(32, 2)
        self.sft = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.fc3(x)
        x = nn.ReLU()(x)
        x = self.fc4(x)
        x = self.sft(x)
        return x


class SyNet_server_bank_5(nn.Module):
    def __init__(self):
        super(SyNet_server_bank_5, self).__init__()
        self.fc1 = torch.nn.Linear(128, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 32)
        self.fc4 = torch.nn.Linear(32, 16)
        self.fc5 = torch.nn.Linear(16, 2)
        self.sft = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.fc3(x)
        x = nn.ReLU()(x)
        x = self.fc4(x)
        x = nn.ReLU()(x)
        x = self.fc5(x)
        x = self.sft(x)
        return x

class SyNet_client_gtsrb(nn.Module):
    def __init__(self,n_workers=2):
        super(SyNet_client_gtsrb, self).__init__()
        self.n_workers = n_workers
        self.features_1 = nn.Sequential(
            # 1
            conv3x3(3, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 2
            conv3x3(64, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 3
            conv3x3(64, int(128)),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.features_2 = nn.Sequential(
            # 4
            conv3x3(128, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 5
            conv3x3(128, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 6
            conv3x3(256, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 7
            conv3x3(256, 256),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.features_3 = nn.Sequential(
            # 8
            conv3x3(256, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 9
            conv3x3(256, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 10
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 11
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.features_4 = nn.Sequential(
            #12
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 13
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 14
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 15
            conv3x3(512, 1024),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            # 16
            conv3x3(1024, 1024),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )
        self.features_5 = nn.Sequential(
            # 4
            conv3x3(128, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 5
            conv3x3(128, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 6
            conv3x3(256, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 7
            conv3x3(256, 128),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.classifier = nn.Sequential(
            # 17
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Dropout(),
            # 18
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            # 19
            nn.Linear(4096, int(128/n_workers)),
        )

    def forward(self, x):
        out = self.features_1(x)
        if self.n_workers<16:
            out = self.features_2(out)
        if self.n_workers<8:
            out = self.features_3(out)
        if self.n_workers<4:
            out = self.features_4(out)
        if self.n_workers==16:
            out = self.features_5(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

class SyNet_server_gtsrb(nn.Module):
    def __init__(self, upload_method=0):
        super(SyNet_server_gtsrb, self).__init__()
        if upload_method == 0:
            self.fc = torch.nn.Linear(128, 43)
        elif upload_method > 0:
            self.fc = torch.nn.Linear(64, 43)
        self.sft = nn.LogSoftmax(dim=1)
        self._sft = nn.Softmax(dim=1)

    def forward(self, x, not_log=False):
        x = self.fc(x)
        if not_log:
            x = self._sft(x)
        else:
            x = self.sft(x)
        return x


class SyNet_server_gtsrb_2(nn.Module):
    def __init__(self):
        super(SyNet_server_gtsrb_2, self).__init__()
        self.fc1 = torch.nn.Linear(128, 128)
        self.fc2 = torch.nn.Linear(128, 43)
        self.sft = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = self.sft(x)
        return x


class SyNet_server_gtsrb_3(nn.Module):
    def __init__(self):
        super(SyNet_server_gtsrb_3, self).__init__()
        self.fc1 = torch.nn.Linear(128, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 43)
        self.sft = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.fc3(x)
        x = self.sft(x)
        return x


class SyNet_server_gtsrb_4(nn.Module):
    def __init__(self):
        super(SyNet_server_gtsrb_4, self).__init__()
        self.fc1 = torch.nn.Linear(128, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 32)
        self.fc4 = torch.nn.Linear(32, 43)
        self.sft = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.fc3(x)
        x = nn.ReLU()(x)
        x = self.fc4(x)
        x = self.sft(x)
        return x


class SyNet_server_gtsrb_5(nn.Module):
    def __init__(self):
        super(SyNet_server_gtsrb_5, self).__init__()
        self.fc1 = torch.nn.Linear(128, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, 64)
        self.fc4 = torch.nn.Linear(64, 32)
        self.fc5 = torch.nn.Linear(32, 43)
        self.sft = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.fc3(x)
        x = nn.ReLU()(x)
        x = self.fc4(x)
        x = nn.ReLU()(x)
        x = self.fc5(x)
        x = self.sft(x)
        return x 


