class Arguments:
    def __init__(self):
        self.near = 2.
        self.far = 6.
        #
        # 粗糙网络参数
        self.netdepth = 8
        self.netwidth = 256
        #
        # 精细网络参数
        self.netdepth_fine = 8
        self.netwidth_fine = 256
        #
        # 每个step 随机射线数
        # 32*32*4 == 4096
        self.N_rand = 32*32*4
        self.netwidth = 256

        self.lrate = 5e-4
        self.lrate_decay = 250
        self.chunk = 1024 * 32
        self.netchunk = 1024 * 64
        self.N_samples = 64
        self.N_importance = 128
        self.perturb = 1
        #
        # 本质为 是否使用 简化设计 为True则不使用
        self.use_viewdirs = True
        #
        # 控制get_embedder() 为0则使用位置编码 为-1不使用并返回恒等函数和3(输入维度, 正常为63)
        self.i_embed = 0
        #
        # 位置编码时 空间坐标
        # 用于生成2的各个次方
        # 1. get_embedder(args.multires, args.i_embed)
        # 2. 'max_freq_log2' : multires - 1,
        #    'num_freqs' : multires,
        # 3. max_freq = self.kwargs['max_freq_log2']
        #    N_freqs = self.kwargs['num_freqs']
        # 4. freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        self.multires = 10
        #
        # 位置编码时 观察方向使用的参数
        # get_embedder(args.multires_views, args.i_embed)
        self.multires_views = 4
        #
        #
        self.raw_noise_std = 0
        #
        #
        self.white_bkgd = True
        self.no_ndc = True
        self.i_weights = 10000
        self.i_video = 50000
        self.i_print = 100
        self.expname = "lego"
        self.basedir = "./logs/"
        self.modeldir = './model/'

        self.ft_path = "None"
