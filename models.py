# %%
import torch
from torch import nn
from model import MambaConfig, MambaBlock, Mamba
import numpy as np
# torch.set_default_device("mps")
# %%

class Res(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1001, 1024, 3, 2)
        self.conv2 = nn.Linear(1024, 2048)

    def forward(self, x):
        return self.conv2(self.conv1(x))

class main_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(80, 2048)
        self.blocks = Mamba(MambaConfig(2048, 6, 2))
        # self.head = nn.Linear(1024, 256)
        self.fhead = nn.Linear(2048, 1024)
        self.head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        # self.head.weight = self.embed.weight

    def forward(self, x:torch.Tensor):
        x = self.embed(x.transpose(2, 1))
        x = self.blocks(x)
        return self.head(self.fhead(x[:, -1]))
    
# # %%
# model = main_model()
# model.eval()
# model(torch.rand((1, 80, 2000)))
# # model(torch.randint(0, 256, (2, 768)))
# # %%
# a = [torch.arange(i, i+100, dtype=torch.int32) for i in range(100)]
# data = torch.tensor(np.array(a))
# data.shape

# # %%
# loss_fn = nn.CrossEntropyLoss(reduction="none")
# opt = torch.optim.Adam(model.parameters())
# # %%
# def trainLoop(data):
#     model.train(True)
#     batch = 10
#     for patch in range(0, len(data)-(batch), batch):
#         opt.zero_grad()
#         y_ = model.forward(data[patch:patch+batch])
#         # print(y_.shape)
#         # print(data[patch+1 : patch + 6].to(torch.int32).shape)
#         loss = loss_fn(torch.transpose(y_, 1, 2), data[patch+1 : patch + batch+1].to(torch.long))
#         loss.sum().backward()
#         opt.step()
#         print(loss.sum().item())

#     model.eval()
# # %%
# for i in range(100):
#     trainLoop(data)
# # %%
# pre = model(data[99:100])
# # %%
# torch.argmax(pre, -1)
# # %%
# data[99]