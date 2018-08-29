# 用Pytorch实现WGAN

---

本文是[解读WGAN](https://alexshuang.github.io/2018/08/24/%E8%A7%A3%E8%AF%BBWasserstein-GAN/)的实践篇，目标是用pytorch实现能生成人脸图像的WGAN。如果对WGAN、DCGANs和GANs还不熟悉的话，可以先阅读[解读WGAN](https://alexshuang.github.io/2018/08/24/%E8%A7%A3%E8%AF%BBWasserstein-GAN/)这篇理论博文，本文不再详解其原理架构，完整的源代码请查看[https://github.com/alexshuang/wgan-pytorch](https://github.com/alexshuang/wgan-pytorch)。

## Looking at the code
[Notebook](https://github.com/alexshuang/wgan-pytorch/wgan-pytorch.ipynb) / [Paper](https://arxiv.org/abs/1701.07875)

我们知道GANs可以用来做很多事情，像生成音频、文字、图像、视频等，本文目标就是用GANs来生成人脸图像，我们将会用到[CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)，一个提供名人头像的数据集。

### Dataset
一个神经网络模型主要由Dataset、Model和Loss function这三部分构成，首先我们先看Dataset：

你可以通过CelebA官网上的指引来下载数据集，当然也可以像我这样使用[kaggle](http://www.kaggle.com)的数据集：
```kaggle datasets download -d jessicali9530/celeba-dataset```

我是基于[fast.ai](https://github.com/fastai/fastai)深度学习框架来编写程序的，它为加载dataset提供了从csv文件加载数据的API，所以我们需要先生成dataset的metadata--files.csv：
```
files = PATH.glob('**/*.jpg')
with CSV_PATH.open('w') as fp:
  for f in files: fp.write(f'{f.relative_to(IMG_PATH)},0\n')
```
因为GANs是不需要指定label的，所以filename后面跟着的是0。
```
bs = 64
sz = 64

tfms = tfms_from_stats(inception_stats, sz)
md = ImageClassifierData.from_csv(PATH, TRN_DN, CSV_PATH, bs, tfms=tfms, skip_header=False, continuous=True)
```
ImageClassifierData.from_csv()就是上文所提到的“从csv文件加载数据的API”。ImageClassifierData是继承于pytorch dataset的子类，通过trn_dl、val_dl、test_dl这三个迭代器就可以获取相关数据集的样本数据。
```
x, _ = next(iter(md.trn_dl))
x_ims = md.trn_ds.denorm(x)

fig, axes = plt.subplots(3, 4, figsize=(6, 4))
for i, ax in enumerate(axes.flat):
  ax.imshow(x_ims[i])
plt.tight_layout()
```
![tou.png](https://upload-images.jianshu.io/upload_images/13575947-166f2b30a1c0750d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
可以看出数据集是64x64大小的头像，WGAN适用此类低分辨率的图像，而高分辨率的图像则更需要像SRWGAN这类模型，后续有机会我也会继续分享相关模型在不同数据集上的应用。

---

### Model
WGAN的作者实现了分别基于[GANs](https://arxiv.org/abs/1406.2661)和[DCGANs](https://arxiv.org/abs/1511.06434)的两个版本，我这里采用DCGANs的网络架构。

##### DCGANs的实现细节：
![image.png](https://upload-images.jianshu.io/upload_images/13575947-10d3ded17655d617.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
除此之外，paper里还提到几个细节：
- generator的输入是长度为100的向量，输出的是rank 4，64x64大小的matrix。
- 不要将batchnorm部署在generator的输出层和discriminator的输入层。
- LeakyRelu的"slope of the leak"系数设定为0.2。
##### Discriminator
```
class ConvBlock(nn.Module):
  def __init__(self, in_c, out_c, ks, stride, bn=True):
    super().__init__()
    padding = ks // 2 // stride
    self.conv = nn.Conv2d(in_c, out_c, ks, stride, padding, bias=False)
    self.bn = nn.BatchNorm2d(out_c) if bn else None
    self.relu = nn.LeakyReLU(0.2, inplace=True)
  
  def forward(self, x):
    x = self.relu(self.conv(x))
    return self.bn(x) if self.bn else x


class DCGAN_D(nn.Module):
  def __init__(self, sz, in_c, nf, num_extra_layers=0):
    super().__init__()
    assert sz % 16 == 0, 'sz must be a multipe by 16.'
   
    self.initial = ConvBlock(in_c, nf, 4, 2, bn=False)
    grid_sz = sz / 2
    self.extra = nn.Sequential(*[ConvBlock(nf, nf, 3, 1) for o in range(num_extra_layers)])
    
    layers = []
    while grid_sz > 4:
      layers.append(ConvBlock(nf, nf * 2, 4, 2))
      grid_sz /= 2; nf *= 2
    self.pyramid = nn.Sequential(*layers)
    
    self.final = nn.Conv2d(nf, 1, 4, 1, 0, bias=False)
  
  def forward(self, x):
    x = self.initial(x)
    x = self.extra(x)
    x = self.pyramid(x)
    return self.final(x).mean().view(1)
```
discriminator是CNN classifier，最终输出一个scalar，该值越小代表fake data的真实系数越高。这里的代码很多都是从原作者的源码中直接借鉴过来的，它通过多个stride convolution操作直至input的grid size等于4，最终取其均值作为output。通常情况下，CNN classifier会以adaptive average pooling生成rank 4, 1x1大小的matrix，将其做flatten操作后作为输入传入全连接层，或许adaptive average pooling也能应用到discriminator中，而且达到更好的效果。

##### Generator
```
class DeconvBlock(nn.Module):
  def __init__(self, in_c, out_c, ks, stride, pad, bn=True):
    super().__init__()
    self.conv = nn.ConvTranspose2d(in_c, out_c, ks, stride=stride, padding=pad, bias=False)
    self.bn = nn.BatchNorm2d(out_c)
    self.relu = nn.ReLU(inplace=True)
    
  def forward(self, x):
    x = self.relu(self.conv(x))
    return self.bn(x) if bn else x


class DCGAN_G(nn.Module):
  def __init__(self, sz, in_c, nf, num_extra_layers=0):
    super().__init__()
    assert sz % 16 == 0, 'sz must be a multipe by 16.'
    i = 4; nf //= 2
    while i != sz:
      nf *= 2; i *= 2
    
    self.initial = DeconvBlock(in_c, nf, 4, 1, 0)
    
    layers = []
    curr_sz = 4
    while curr_sz < (sz // 2):
      layers.append(DeconvBlock(nf, nf // 2, 4, 2, 1))
      curr_sz *= 2; nf //= 2
    self.pyramid = nn.Sequential(*layers)
    self.extra = nn.Sequential(*[DeconvBlock(nf, nf, 3, 1, 1) for i in range(num_extra_layers)])
    self.final = nn.ConvTranspose2d(nf, 3, 4, 2, 1, bias=False)
  
  def forward(self, x):
    x = self.initial(x)
    x = self.pyramid(x)
    x = self.extra(x)
    return F.tanh(self.final(x))
```
generator和discriminator相反，通过deconvolution将rank 1的向量转化为rank 4, 64x64的矩阵。rank 1向量就是长度为100的噪声向量，为deconvolution()的参数格式为rank 4，因此需要gen_noise()将rank 1的噪声转化为rank 4矩阵。
### Noise Generator
```
def gen_noise(bs):
  return V(torch.zeros((bs, nz, 1, 1)).normal_(0, 1))
```
![image.png](https://upload-images.jianshu.io/upload_images/13575947-681149bcb326eb55.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

---

### WGAN / [paper](https://arxiv.org/abs/1701.07875)
![WGAN.png](https://upload-images.jianshu.io/upload_images/13575947-9ee16a3d12939ab8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
```
ncritic = 5

def WGAN_fit(md, niter):
  net_G.train(True)
  net_D.train(True)
  gen_iterations = 0
  
  for epoch in range(niter):
    data_iter = iter(md.trn_dl)
    i, n = 0, len(md.trn_dl)
    with tqdm(total=n) as pbar:
      while i < n:
        set_trainable(net_D, True)
        set_trainable(net_G, False)
        j = 0
        while i < n and j < ncritic:
          for p in net_D.parameters(): p.data.clamp_(-0.01, 0.01)
          real = V(next(data_iter)[0])
          real_loss = net_D(real)
          fake = net_G(gen_noise(real.size(0)))
          fake_loss = net_D(V(fake.data))
          net_D.zero_grad()
          d_loss = real_loss - fake_loss
          d_loss.backward()
          opt_D.step()
          i += 1; j += 1
          pbar.update()
        
        set_trainable(net_D, False)
        set_trainable(net_G, True)
        net_G.zero_grad()
        g_loss = net_D(net_G(gen_noise(bs)))
        g_loss.backward()
        opt_G.step()
        gen_iterations += 1
      
    print(f'd_loss: {to_np(d_loss)}; g_loss: {to_np(g_loss)}; '
          f'real_loss: {to_np(real_loss)}; fake_loss: {to_np(fake_loss)}')
```
paper里已经给出了明确的实现方法以及各超参系数，这里不需要额外多说什么了，如果你对其训练方法或涉及到的公式有所不解，请点击[解读WGAN](https://alexshuang.github.io/2018/08/24/%E8%A7%A3%E8%AF%BBWasserstein-GAN/)。有经验的朋友看到这会觉得很奇怪，为什么这里没有metric方法？GANs是无监督学习的模型，所以大家都会调侃GANs是一种不用看metric指标的模型哈哈。

---

### 模型训练
```
files = PATH.glob('**/*.jpg')
with CSV_SMALL_PATH.open('w') as fp:
  for f in files:
    if np.random.random() < 0.1: fp.write(f'{f.relative_to(IMG_PATH)},0\n')
```
为了快速验证模型的有效性，先用10%的数据集来训练。
```
lr = 5e-5
opt_G = optim.RMSprop(net_G.parameters(), lr=lr)
opt_D = optim.RMSprop(net_D.parameters(), lr=lr)
WGAN_fit(md, 4)
```
```
100%|██████████| 256/256 [01:12<00:00,  1.43it/s]
d_loss: [-1.06658]; g_loss: [0.81221]; real_loss: [-0.27118]; fake_loss: [0.7954]
100%|██████████| 256/256 [01:09<00:00,  3.91it/s]
d_loss: [-1.09587]; g_loss: [0.84473]; real_loss: [-0.26713]; fake_loss: [0.82875]
100%|██████████| 256/256 [01:10<00:00,  3.89it/s]
d_loss: [-1.04561]; g_loss: [0.82418]; real_loss: [-0.23558]; fake_loss: [0.81003]
100%|██████████| 256/256 [01:09<00:00,  3.92it/s]
d_loss: [-1.06523]; g_loss: [0.8402]; real_loss: [-0.24339]; fake_loss: [0.82184]
100%|██████████| 256/256 [01:10<00:00,  3.89it/s]
d_loss: [-1.01388]; g_loss: [0.82767]; real_loss: [-0.22356]; fake_loss: [0.79032]
```
衡量GANs能力强弱，看fake_loss即可，假设discriminator的鉴别能力会随着generator的造假能力增强而增强，那discriminator的输出--fake_loss，其值越小证明真实程度越高，反之亦然。我们将输出显示出来：
```
def gallery(x, nc=3):
    n,h,w,c = x.shape
    nr = n//nc
    assert n == nr*nc
    return (x.reshape(nr, nc, h, w, c)
              .swapaxes(1,2)
              .reshape(h*nr, w*nc, c))

fixed_noise = gen_noise(bs)

net_D.eval(); net_G.eval();
fake = net_G(fixed_noise).data.cpu()
faked = np.clip(md.trn_ds.denorm(fake),0,1)

plt.figure(figsize=(9,9))
plt.imshow(gallery(faked, 8))
```
![image.png](https://upload-images.jianshu.io/upload_images/13575947-fa6484760af0e239.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
我们看到输出结果很模糊，只有一些轮廓。接着我们再用完整的数据集来训练。
```
files = PATH.glob('**/*.jpg')
with CSV_PATH.open('w') as fp:
  for f in files: fp.write(f'{f.relative_to(IMG_PATH)},0\n')
```
```
100%|██████████| 2533/2533 [07:16<00:00,  7.29it/s]
d_loss: [-0.71795]; g_loss: [0.25149]; real_loss: [-0.2049]; fake_loss: [0.51305]
100%|██████████| 2533/2533 [07:17<00:00,  7.24it/s]
d_loss: [-0.66687]; g_loss: [0.22367]; real_loss: [-0.27482]; fake_loss: [0.39205]
100%|██████████| 2533/2533 [07:15<00:00,  7.34it/s]
d_loss: [-0.62798]; g_loss: [0.41359]; real_loss: [-0.35229]; fake_loss: [0.27569]
100%|██████████| 2533/2533 [07:15<00:00,  7.24it/s]
d_loss: [-0.43533]; g_loss: [0.1597]; real_loss: [-0.01642]; fake_loss: [0.41891]
100%|██████████| 2533/2533 [07:16<00:00,  7.28it/s]
d_loss: [-0.4477]; g_loss: [0.42561]; real_loss: [-0.48342]; fake_loss: [-0.03572]
100%|██████████| 2533/2533 [07:17<00:00,  7.26it/s]
```
可以看到现在的训练样本是之前的10倍，fake_loss也相比之前要低很多，我们也将输出结果显示出来：
![image.png](https://upload-images.jianshu.io/upload_images/13575947-e600691a72b0ea59.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
可以看到，虽然只训练了5个epoch，但已经可以看出清晰的人脸，虽然有些人脸很诡异，但鉴于训练时间较短，模型表现上还是OK的。

---

## 引用
[DCGANs](https://arxiv.org/abs/1511.06434)
[WGAN](https://arxiv.org/abs/1701.07875)
[fasiai](https://github.com/fastai/fastai/blob/master/courses/dl2/wgan.ipynb)

