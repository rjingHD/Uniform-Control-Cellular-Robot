def add_arguments(parser):
    '''
    Add your arguments here if needed. The TA will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate for training')
    parser.add_argument('--replay_buffer_size', type=int, default=500000, help='replay buffer size for training')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--img_height', type=int, default=3, help='image height for training')
    parser.add_argument('--img_width', type=int, default=11, help='image width for training')
    parser.add_argument('--img_channel', type=int, default=1, help='number of images in one observation for training')
    parser.add_argument('--load_model_training', type=bool, default=False, help='load model while training')
    return parser
