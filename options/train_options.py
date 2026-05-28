from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--display_freq', type=int, default=2360, help='frequency of showing training results on screen')
        self.parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=100, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=4, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--continue_train', default=False,help='continue training: load the latest model')
        self.parser.add_argument('--epoch_count', type=int, default=0, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='0', help='which epoch to load set to latest to use latest cached model')
        self.parser.add_argument('--niter', type=int, default=200, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=200, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--lambda_A', type=float, default=0, help='L1_img')
        self.parser.add_argument('--lambda_B', type=float, default=0, help='distribution')
        self.parser.add_argument('--lambda_C', type=float, default=0, help='GAN_loss')
        self.parser.add_argument('--lambda_D', type=float, default=1, help='loss_B_mean')
        self.parser.add_argument('--lambda_E', type=float, default=15, help='B_smooth')
        self.parser.add_argument('--lambda_F', type=float, default=0.3, help='SegmentUniformL')
        self.parser.add_argument('--lambda_I', type=float, default=0.3, help='TNLinfer')
        self.parser.add_argument('--lambda_H', type=float, default=0, help='ABCnetPL')
        self.parser.add_argument('--identity', type=float, default=0.0, help='use identity mapping. Setting identity other than 1 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set optidentity = 0.1')
        self.parser.add_argument('--pool_size', type=int, default=20, help='the size of image buffer that stores previously generated images')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        self.parser.add_argument('--display_env', type=str, default='lap_lr', help='visdom display environment name (default is "main")')
        self.isTrain = True
