import imageio
import numpy as np
a = imageio.imread("data/nerf_synthetic/multihuman/render/012_mask.png")
imgs=[]
x,y,z=np.nonzero(a)
print(len(x)/3)
for i in range (100):
    imgs.append(a)

imgs = (np.array(imgs) / 255.).astype(np.float32)
b = np.nonzero(imgs)
print(b[0].shape)

print(np.where(imgs==0))

import torch
a = [[1,2],[2,3],[3,4]]
b =[[1,2]]
c= [x for x in a if x not in b]
print("c:", torch.tensor(c))
print((torch.tensor(c)).shape)
b = torch.tensor([[1,2]])
b1 = b.numpy().tolist()
print(b1)
d = []
d.append([])
d[0].append(1)
d.append([])
d[1].append(2)
print(d)


a = 2.4
print("a:",int(np.ceil(a)))


x = torch.tensor([[1,2,3],[3,4,5]])
print(x.shape[:-1])