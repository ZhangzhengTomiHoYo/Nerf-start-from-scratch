import torch
from tqdm import tqdm
from arguments import Arguments
from loadData import get_rays_rgb
from nerf import create_nerf
import time
if __name__=="__main__":

    args = Arguments()
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near' : args.near,
        'far' : args.far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    N_rand = args.N_rand

    N_iters = 2000000 + 1

    i_train = 100
    print('Begin')
    print('Train views are', i_train)
    print('Test views are', 0)
    print('val views are', 0)

    rays_rgb = get_rays_rgb()

    i_batch = 0
    start = start + 1
    for i in tqdm(range(start, N_iters)):
        time0 = time.time()
        batch = rays_rgb[i_batch:i_batch+N_rand]
        batch = torch.transpose(batch, 0, 1)
        #
        # 我的理解
        # batch_rays 训练数据
        # target_s   训练标签
        batch_rays, target_s = batch[:2], batch[2]

        i_batch += N_rand
        if i_batch >= rays_rgb.shape[0]:
            print('')
            rand_idx = torch.randperm(rays_rgb.shape[0])
            rays_rgb = rays_rgb[rand_idx]
            i_batch = 0

        ###### 渲染 ######
        rgb, disp, acc, extras = render(H, W, K, chunk=)
