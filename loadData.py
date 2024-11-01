import os, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm, trange

np.random.seed(0)

splits = ['train', 'val', 'test']

basedir = '.\data\lego'

metas = {}
for s in splits:
    with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
        metas[s] = json.load(fp)

all_imgs = []
all_poses = []
counts = [0]

for s in splits:
    meta = metas[s]
    imgs = []
    poses = []
    for frame in meta['frames'][::]:
        fname = os.path.join(basedir, frame['file_path'] + '.png')
        imgs.append(imageio.imread(fname))
        poses.append(np.array(frame['transform_matrix']))

    imgs = (np.array(imgs)/ 255.).astype(np.float32)
    print("imgs shape:", imgs.shape)
    poses = np.array(poses).astype(np.float32)

    counts.append(counts[-1] + imgs.shape[0])
    all_imgs.append(imgs)
    all_poses.append(poses)

imgs = np.concatenate(all_imgs, 0)
poses = np.concatenate(all_poses, 0)
print("imgs shape:", imgs.shape)
print("poses shape:", poses.shape)
# imgs shape: (400, 800, 800, 4) 训练集 100张 验证集100张 测试机200张 # RGBA -> RGB + alpha透明度通道
# poses shape: (400, 4, 4)

# plt.figure('lego')
# plt.imshow(imgs[10])
# plt.title('Lego Image')
# plt.show()

camera_angle_x = float(meta['camera_angle_x'])

H, W = imgs[0].shape[:2]

focal = .5 * W / np.tan(.5 * camera_angle_x)

K = np.array([
    [focal, 0, 0.5*W],
    [0, focal, 0.5*H],
    [0, 0, 1]
])
print("K shape:", K.shape)
def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))

    # i 800*800, j 800*800, K[0][2] 800*800, K[0][0] 800*800
    # ->
    # dirs 800*800*3
    dirs = np.stack([i-K[0][2]/K[0][0], -(j-K[1][2]/K[1][1]), -np.ones_like(i)], -1)

    # ... 代表选了前两个维度
    # np.newaxis 直接增加一个维度
    #  :
    # dirs 800*800*3 -> 800*800*1*3
    # c2w 是从 poses 裁剪过来的 : poses 400*4*4 -> c2w 3*4 -> c2w[:3,:3] 3*3 这样乘法做广播就很明显了
    # ->
    # 800*800*3*3
    # rays_d after sum axis=-1 -> 800*800*3
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)

    # c2w[:3, -1] 注意是-1 不带: 这个轴/维度会消失 因此 (3,)
    # 800*800*3
    # ->
    # rays_o 800*800*3
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))

    return rays_o, rays_d
                                         # 取循环的时候会把第0轴给去掉 poses 400*4*4 -> p 3*4
rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # rays 400*2*800*800*3
print("rays shape:", rays.shape)

# (400, 800, 800, 4) -> (400, 800, 800, 3)
imgs = imgs[..., :3]
                                       # imgs[:, None] 增加了一个新的维度 None 在这里相当于 np.newaxis
rays_rgb = np.concatenate([rays, imgs[:,None]], 1)
print("rays_rgb shape after concatenate:", rays_rgb.shape)

rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])
print("rays_rgb shape after transpose:", rays_rgb.shape)

# train文件夹中有100张图片
i_train = list(range(100))
rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0)
print("rays_rgb shape after stack:", rays_rgb.shape)

rays_rgb = np.reshape(rays_rgb, [-1,3,3])
print("rays_rgb shape after reshape:", rays_rgb.shape)

rays_rgb = rays_rgb.astype(np.float32)
print("rays_rgb shape after astype:", rays_rgb.shape)

np.random.shuffle(rays_rgb)
print("rays_rgb shape after shuffle:", rays_rgb.shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

imgs = torch.Tensor(imgs).to(device)
print("imgs shape after to(device):", imgs.shape)

poses = torch.Tensor(poses).to(device)
print("poses shape after to(device):", poses.shape)

rays_rgb = torch.Tensor(rays_rgb).to(device)
print("rays_rgb shape after to(device):", rays_rgb.shape)


'''
imgs shape: (400, 800, 800, 4)
poses shape: (400, 4, 4)
K shape: (3, 3)
rays shape: (400, 2, 800, 800, 3)
rays_rgb shape after concatenate: (400, 3, 800, 800, 3)
rays_rgb shape after transpose: (400, 800, 800, 3, 3)
rays_rgb shape after stack: (100, 800, 800, 3, 3)
rays_rgb shape after reshape: (64000000, 3, 3)
rays_rgb shape after astype: (64000000, 3, 3)
rays_rgb shape after shuffle: (64000000, 3, 3)
imgs shape after to(device): torch.Size([400, 800, 800, 3])
poses shape after to(device): torch.Size([400, 4, 4])
rays_rgb shape after to(device): torch.Size([64000000, 3, 3])
'''

