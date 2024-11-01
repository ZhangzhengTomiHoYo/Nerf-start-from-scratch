import os

import torch
from torch import nn
import torch.nn.functional as F
from arguments import Arguments, Arguments

from Embedder import get_embedder

device = 'cuda'
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        #
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)]
        )

        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                #
                # TODO 疑惑？ cat后形状不会变吗 怎么还能正常前向
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            #
            # 第9层
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
            #
            # 第10层
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)
            rgb = self.rgb_linear(h)

            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs
    
def create_nerf(args):
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    #
    # alpha 1维
    # color 3维
    # 共4维
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    #
    # 创建粗糙网络
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    #
    # 创建精细网络
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netwidth_fine, W=args.netwidth_fine,
                     input_ch=input_ch, output_ch=output_ch, skips=skips,
                     input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())

    #
    # 创建优化器
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))
    start = 0
    basedir = args.basedir
    expname= args.expname
    #
    # 可以通过增加arguments 进行checkpoint
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        #
        """
        这个代码的逻辑顺序是：
        1. 后面的os.path.join(basedir, expname)：拼接出目录路径
        2. basedir / expname，即指定的实验文件夹路径。
        3. os.listdir(os.path.join(basedir, expname))：列出指定路径下所有的文件和文件夹。
        4. sorted(...)：对列出的文件进行排序，默认按字母顺序。
        4. [f for f in ... if 'tar' in f]：筛选出包含“tar”字符串的文件。
        6. 前面的os.path.join(basedir, expname, f)：将文件名与目录路径结合，生成每个符合条件文件的完整路径。
        最终，ckpts变量将存储所有符合条件的文件路径（按字母顺序排列），这些文件名中包含“tar”。"""
        ckpts = [os.path.join(basedir,expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print("找到checkpoints", ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)
        #
        # global_step 全局训练步数
        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        #
        # 加载模型
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##############
    render_kwargs_train = {
        # 'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_inportance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer

if __name__ == "__main__":
    args = Arguments()
    create_nerf(args)
