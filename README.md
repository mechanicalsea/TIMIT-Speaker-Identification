# TIMIT Speaker Identification

Speaker Identification in TIMIT dataset.

There are several experimental results under various settings. These results may give insight for other datasets.

This is only a document. No Coding.

## Basis Step

1. load waveforms and extract spectogram,
2. define deep convolutional network and optimizer,
3. prepare trainset and testset,
4. train network.

Note that iterate model for improving.

## Main Conclusinon

Note: just for TIMIT-like dataset that is clean and ideal small ones.

1. training on short-time utterances (1s) benefit results and improve 80+% to 90+%.
2. decreasing or increasing amplitude of signals within $\pm 0.2$ benefit results and improve 80+% to 95+%.
   
    - This operation is inspired by sincnet settings.
    - **It can be replaced by dB-psd plused 110.**
    
3. SGD is more stable than Adam.
4. well-investigated netowrk is with 3 conv2d and 512-512 hidden units.
5. power spectral density in dB $10\log_{10}(\text{spec})$ gives best inputs.
6. The range of feature of inputs contributes performance, and results showed that $\text{magnitude}\in[0, 60]$, $\text{psd in dB}\in[-60, 40]$, and $\text{psd in dB plused 110 of normalized waveform}\in[-40, 60]$ benefit the learning process.
7. Hypothesis (verified): seperate distribution of values of inputs may harm performance, compared to dense/continous distribution.

## Results Overview

### 1. On `TIMIT_TRAIN.txt` (5 utterances) and `TIMIT_TEST.txt` (3 utterances) 462 speaker

|  ID  | file                   | top 1 acc | top 5 acc | train loss | note                                                         |
| :--: | :--------------------- | --------: | --------: | ---------: | :----------------------------------------------------------- |
|  1   | `462-amp0.2`           |   `0.965` |   `0.993` |    `0.051` | decrease or increase 0.2 amplitude of signals                |
|  2   | `462-amp0.2-normalize` |   `0.731` |   `0.914` |    `0.094` | `ID 1` - then normalize it                                   |
|  3   | ``462-normalize-5_lr`` |   `0.780` |   `0.935` |    `0.173` | normalize it with decreasing learning rate at every 5 epochs |

### 2. On `iden_train.txt` (8 utterances) and `iden_test.txt` (2 utterances) 630 speakers

|    ID     | file                       | top 1 acc | top 5 acc | train loss | note                                              |
| :-------: | :------------------------- | --------: | --------: | ---------: | :------------------------------------------------ |
|     1     | `630-amp0.2`               |   `0.995` |   `0.999` |    `0.019` | decrease or increase 0.2 amplitude of signals     |
|     2     | `630-amp0.2-small`         |   `0.993` |   `0.999` |    `0.033` | `ID 1` - decrease network's size                  |
|     3     | `630-amp0.2-tiny`          |   `0.984` |   `0.998` |    `0.044` | `ID 1` - decrease network's size                  |
|     4     | `630-amp0.2-tiny-adam`     |   `0.971` |   `0.994` |    `0.053` | `ID 3` - use Adam optimizer                       |
|     5     | `630-amp0.2-footnote-adam` |   `0.983` |       `1` |     `0.01` | `ID 3` - reduce the size and the stride of conv2d |
| 6$^\star$ | `630-amp0.2-footnote-sgd`  |   `0.991` |   `0.998` |    `0.065` | `ID 5` - replace optimizer SGD                    |
| 7$^\star$ | `630-amp0.2-onesided`      |    `0.99` |       `1` |    `0.062` | `ID 6` - only one side spectrum                   |

### 3. No amplify $\pm 0.2$ On `iden_train.txt` (8 utterances) and `iden_test.txt` (2 utterances) 630 speakers

|    ID     | file                                    | top 1 acc | top 5 acc | train loss | note                                                         |
| :-------: | :-------------------------------------- | --------: | --------: | ---------: | :----------------------------------------------------------- |
|     1     | None                                    |  `0.8278` |  `0.9429` |  `0.02518` | 3s + dither + preemphasis (30 epochs)                        |
|     2     | None                                    |  `0.8579` |  `0.9532` |   `0.0252` | 3s + preemphasis (30 epochs)                                 |
|     3     | None                                    |  `0.8278` |  `0.9468` |  `0.02813` | 3s (30 epochs)                                               |
|     4     | None                                    |  `0.9381` |  `0.9865` |  `0.06056` | 1s + preemphasis (100 epochs)                                |
|     5     | None                                    |  `0.8746` |  `0.9706` |  `0.04939` | 0.5s + preemphasis (200 epochs)                              |
| 6$^\star$ | `630-onesided-psd-db-plus110-lr0.01`    |   `0.994` |   `0.999` |   `0.0274` | 1s + s1 + psd in dB and plus 110 using SGD with learning rate 0.01 |
| 7$^\star$ | `630-onesided-s1-psd-dB-plus110-lr0.01` |   `0.997` |       `1` |   `0.0260` | `ID 6` - verify again (run again)                            |

### 4. Spectrum selection 630 speakers

|    ID     | file                                   | top 1 acc | top 5 acc | train loss | note                                                         |
| :-------: | :------------------------------------- | --------: | --------: | ---------: | :----------------------------------------------------------- |
|     1     | `630-amp0.2-onesided-mag`              |  `0.7135` |  `0.9063` |    `3.283` | magnitude: using magnitude as features of inputs, $\text{magnitude}\in[0, 80]$ |
|     2     | `630-amp0.2-onesided-mag-lr0.01`       |  `0.9921` |       `1` |   `0.0664` | `ID 1` - increase initial learning rate 0.01 in SGD          |
|     3     | `630-amp0.2-onesided-psd`              |   `0.180` |   `0.431` |    `4.908` | psd: using psd as features of inputs, generally, $\text{psd} = \text{magnitude}^2, \text{psd}\in[0,8000]$ |
|     4     | `630-amp0.2-onesided-psd-lr0.01`       |   `0.473` |   `0.727` |    `0.541` | `ID 3` - increase initial learning rate 0.01 in SGD          |
|     5     | `630-amp0.2-onesided-psd^0.5-lr0.01`   |   `0.989` |   `0.998` |   `0.0505` | `ID 3` - $\sqrt{\text{spec}}$                                |
| 6$^\star$ | `630-amp0.2-onesided-psd-dB-lr0.01`    |   `0.995` |       `1` |   `0.0387` | `ID 3` - (**fastest in convergence: before 50 epoches**) $10\log_{10}(\text{spec})$ |
|     7     | `630-amp0.2-onesided-psd-norm1-lr0.01` |   `0.183` |   `0.364` |    `0.626` | `ID 3` - normalize $\text{spectrum}\in[0,1]$ or so           |

### 5. Normalize waveform by $2^{15}$ ($32768$ or `int16`) 630 speakers

|    ID     | file                                                 | top 1 acc | top 5 acc | train loss | note                                                         |
| :-------: | :--------------------------------------------------- | --------: | --------: | ---------: | :----------------------------------------------------------- |
|     1     | `630-amp0.2-onesided-s1`                             | `0.00159` |  `0.0119` |    `3.287` | baseline                                                     |
|     2     | `630-amp0.2-onesided-s1log10`                        |   `0.985` |   `0.998` |   `0.4034` | `ID 1` - $\log_{10}(spec)$                                   |
|     3     | `630-amp0.2-onesided-s1log10-RMSprop`                |   `0.982` |   `0.999` | `0.000191` | `ID 1` - (unstable) replace SGD to RMSprop                   |
|     4     | `630-amp0.2-onesided-s1log10-psd-dB-lr0.01`          |   `0.994` |       `1` |   `0.0230` | `ID 1` - (a bit unstable) using psd in dB as inputs and SGD starting with 0.01 learning rate |
| 5$^\star$ | `630-amp0.2-onesided-s1-psd-dB-plus110-lr0.01`       |   `0.995` |   `0.999` |   `0.0348` | `ID 1` - (**Selected**: **fastest in convergence: before 50 epoches**) plus psd in dB 110 for rearrange spectral values in $[-50, 50]$ or so |
|     6     | `630-amp0.2-onesided-s1-psd-dB-plus110-norm1-lr0.01` |   `0.997` |   `0.999` |   `0.0446` | `ID 1` - (comparative to **Selected**) normalize $\text{spec} \in [-1, 1]$ or so |
|     7     | `630-amp0.2-onesided-s1-psd-dB-plus110-norm1-lr0.01` |   `0.996` |   `0.999` |   `0.0316` | `ID 1` - (comparative to **Selected**) normalize $\text{spec} \in [-8000, 8000]$ or so |

## Core method

### 1. Data Loader

```python
class IdentificationDataset(Dataset):
    
    def __init__(self, path, lst, transform=None):
        data_path = os.path.join(path, lst)
        self.dataset = pd.read_table(data_path, sep=' ', header=None, 
                                     names=['name', 'path']).reset_index(drop=True).values
        self.speaker_id = {name: i for i, name in enumerate(set(self.dataset[:,0]))}
        self.path = path
        self.train = True if 'train' in data_path.lower() else False
        self.transform = transform
    
    def load_wav(self, audio_path):
        # read .wav
        samples, rate = sf.read(audio_path, dtype='int16')
        samples = samples  / 32768.0
        ## parameters
        window = 'hamming'
        # window width and step size
        Tw = 25 # ms
        Ts = 10 # ms
        # frame duration (samples)
        Nw = int(rate * Tw * 1e-3)
        Ns = int(rate * (Tw - Ts) * 1e-3)
        # overlapped duration (samples)
        # 2 ** to the next pow of 2 of (Nw - 1)
        nfft = 2 ** (Nw - 1).bit_length()
        
        if self.train:
            ''' segment selection and signal amplification 
            '''
            segment_len = int(1.0 * rate)
            if len(samples) <= segment_len:
                samples = np.pad(samples, (0, segment_len + 1 - len(samples)))
            upper_bound = len(samples) - segment_len
            start = np.random.randint(0, upper_bound)
            end = start + segment_len
            samples = samples[start:end]# * np.random.uniform(0.8, 1.2)
        
        # spectogram
        _, _, spec = signal.spectrogram(samples, rate, window, Nw, Ns, nfft, 
                                        mode='psd', return_onesided=True)
        # for numeric consideration
        spec = 10.0 * np.log10(spec) # dB
        spec += 110.0 # plus 110 for psd of normalize waveform 
        
        if self.transform:
            spec = self.transform(spec)
        return spec, rate
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # path
        track_path = self.dataset[idx] # name, path
        audio_path = os.path.join(self.path, track_path[1])
        # read .wav
        spec, _ = self.load_wav(audio_path)
        label = self.speaker_id[track_path[0]]
        return {'label':label, 'spec':spec}
    
class ToTensor(object):
    """Convert spectogram to Tensor."""
    def __call__(self, spec):
        F, T = spec.shape
        # now specs are of size (Freq, Time) and 2D but has to be 3D (channel dim)
        spec = spec.reshape(1, F, T)
        # make the ndarray to be of a proper type (was float64)
        spec = spec.astype(np.float32)
        return torch.from_numpy(spec)
```

### 2. Model Architecture

```python
class VoiceNet(nn.Module):
	"""footnote: 3 convolutional layers
	"""
    def __init__(self, num_classes=2):
        super(VoiceNet, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=60, kernel_size=7, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=60, out_channels=60, kernel_size=5, stride=2, padding=1)
        
        self.bn1 = nn.BatchNorm2d(num_features=60)
        self.bn2 = nn.BatchNorm2d(num_features=60)
        self.bn3 = nn.BatchNorm2d(num_features=512)
        self.bn5 = nn.BatchNorm1d(num_features=512)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Conv2d with weights of size (H, 1) is identical to FC with H weights
        self.conv3 = nn.Conv2d(in_channels=60, out_channels=512, kernel_size=(31, 1))
        self.fc5 = nn.Linear(in_features=512, out_features=512)
        self.fc6 = nn.Linear(in_features=512, out_features=num_classes)
        
    def forward(self, x):
        B, C, H, W = x.size()
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.mpool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        
        _, _, _, W = x.size()
        self.apool4 = nn.AvgPool2d(kernel_size=(1, W)) # average pooling over time
        x = self.apool4(x)
        
        x = x.flatten(start_dim=1)
        x = self.relu(self.bn5(self.fc5(x)))
        x = self.fc6(x)
        
        # during training, there's no need for SoftMax because CELoss calculates it
        if self.training:
            return x
        else:
            return self.softmax(x)
```

## Remarks

Welcome discussion.

Free for questions.
