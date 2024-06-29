# %%
YAML_PATH = "hyperPrams.yaml"


import yaml
with open(YAML_PATH) as f:
    prams = yaml.load(f, Loader=yaml.FullLoader)


from model import Mamba, MambaConfig, HyperPrams, \
                    waveform_load, REAL_DIR, FAKE_DIR, \
                    Process_mel
import torch 
from torch import nn, tensor
import numpy as np

prams = HyperPrams(prams)
process = Process_mel(prams)

from models import main_model
# %%

model = main_model()

# %%
loss_fn = nn.CrossEntropyLoss(reduction="mean")
opt = torch.optim.Adam(model.parameters())
# %%

real_datas = waveform_load(REAL_DIR, prams)
mel_spec_real = process(torch.tensor(real_datas))

fake_datas = waveform_load(FAKE_DIR, prams)
mel_spec_fake = process(torch.tensor(fake_datas))
# %%
datas = torch.concat((tensor(mel_spec_fake), tensor(mel_spec_real)), 0)
datas.shape
# %%
mel_spec_fake.shape, mel_spec_real.shape
# %%
labels = tensor(np.array([0 for i in range(len(mel_spec_fake))] + [1 for i in range(len(mel_spec_real))]))

# %%
mel_spec_real[:1].shape, mel_spec_fake[:1].shape
# %%
import matplotlib.pyplot as plt
# %%
_, ax = plt.subplots(2, 2, figsize = (128, 32))
for i in range(2):
    # left
    ax[i][0].pcolor(mel_spec_real[i])

    # right
    ax[i][1].pcolor(mel_spec_fake[i])
# %%
import gc
del mel_spec_fake, mel_spec_real, real_datas, fake_datas
gc.collect()
# %%
model.to(device)
def trainLoop(data, labels):
    model.train(True)
    batch = 10
    for patch in range(0, len(data), batch):
        opt.zero_grad()
        y_ = model.forward(data[patch:patch+batch].to(device))
        # print(y_.shape)
        # print(data[patch+1 : patch + 6].to(torch.int32).shape)
        loss = loss_fn(y_, labels[patch:patch+batch].to(device))
        loss.backward()
        opt.step()
        print(loss.sum().item())

    model.eval()
# %%
for i in range(10):
    trainLoop(datas, labels)
# %%