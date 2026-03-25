from models import *
import argparse
import os

def get_parser():
    parser = argparse.ArgumentParser(description="Training configuration for onset model")

    parser.add_argument('--stream_labels_fp', type=str, default='onset/songs/stream_labels.pkl',
                        help='Path to the stream labels .pkl file')
    
    parser.add_argument('--shuffle', dest='shuffle', action='store_true',
                        help='Enable data shuffling (default)')
    parser.add_argument('--no-shuffle', dest='shuffle', action='store_false',
                        help='Disable data shuffling')
    parser.set_defaults(shuffle=True)

    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--memlen', type=int, default=15,
                        help='Memory length for audio context')
    parser.add_argument('--mem_size', type=int, default=2500,
                        help='Memory size for windowed observations')
    parser.add_argument('--nframes', type=int, default=32,
                        help='Number of frames per input sliced across beats')
    parser.add_argument('--steps_per_epoch', type=int, default=400,
                        help='Training steps per epoch')
    parser.add_argument('--nepochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--nmelbands', type=int, default=80,
                        help='Number of mel bands in input features')
    parser.add_argument('--nchannels', type=int, default=3,
                        help='Number of channels in input')
    parser.add_argument('--model_dir', type=str, default='trained_models',
                        help='Directory to save trained models')
    parser.add_argument('--train_txt_fp', type=str, default='onset/songs/songs_train.txt',
                        help='Path to training file list')
    parser.add_argument('--test_txt_fp', type=str, default='onset/songs/songs_test.txt',
                        help='Path to testing file list')

    parser.add_argument('--load_checkpoint', dest='load_checkpoint', action='store_true',
                        help='Load from checkpoint (default)')
    parser.add_argument('--no-load_checkpoint', dest='load_checkpoint', action='store_false',
                        help='Do not load from checkpoint')
    parser.set_defaults(load_checkpoint=True)

    parser.add_argument('--full_bidirectional', dest='full_bidirectional', action='store_true',
                        help='Use full bidirectional RNN')
    parser.add_argument('--no-full_bidirectional', dest='full_bidirectional', action='store_false',
                        help='Do not use full bidirectional RNN')
    parser.set_defaults(full_bidirectional=False)

    parser.add_argument('--conv3d', dest='conv3d', action='store_true',
                        help='Use 3D convolution')
    parser.add_argument('--no-conv3d', dest='conv3d', action='store_false',
                        help='Do not use 3D convolution')
    parser.set_defaults(conv3d=False)

    parser.add_argument('--model_name', type=str, default='onset',
                        help='Name of the model')
    
    parser.add_argument('--use_all_charts', action='store_true', default=False, help='Use all charts for song')
    parser.add_argument('--no-use_all_charts', dest='use_all_charts', action='store_false', help='Do not use all charts for songs')

    parser.add_argument('--use_scheduler', action='store_true', default=False, help='Use lr scheduling')
    parser.add_argument('--no-use_scheduler', dest='use_scheduler', action='store_false', help='Do not use lr scheduling')

    parser.add_argument('--use_early_stop', action='store_true', default=False, help='Use early stopping')
    parser.add_argument('--no-use_early_stop', dest='use_early_stop', action='store_false', help='Do not use early stopping')

    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    train_onset_model(stream_labels_fp=args.stream_labels_fp,
                        shuffle = args.shuffle,
                        batch_size = args.batch_size,
                        memlen = args.memlen,
                        mem_size = args.mem_size,
                        nframes = args.nframes,
                        steps_per_epoch = args.steps_per_epoch,
                        nepochs = args.nepochs,
                        nmelbands = args.nmelbands,
                        nchannels = args.nchannels,
                        model_dir = args.model_dir,
                        train_txt_fp = args.train_txt_fp,
                        test_txt_fp = args.test_txt_fp,
                        load_checkpoint = args.load_checkpoint,
                        full_bidirectional = args.full_bidirectional,
                        conv3d = args.conv3d,
                        model_name = args.model_name,
                        use_all_charts= args.use_all_charts,
                        use_scheduler = args.use_scheduler,
                        use_early_stop = args.use_early_stop)
