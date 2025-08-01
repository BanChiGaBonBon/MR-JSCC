from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--output_path', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='JSCCOFDM', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        # parser.add_argument('--num_test', type=int, default=1000, help='how many test images to run')
        parser.add_argument('--how_many_channel', type=int, default=1, help='number of transmission per image')
        parser.add_argument('--fig', type=str, default='', help='')


        # rewrite devalue values
        # parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser
