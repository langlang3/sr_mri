import argparse

def arguments():
    parser = argparse.ArgumentParser(description='MRI_sr')
    
    parser.add_argument('--n_threads', type=int, default=4,
                        help='number of threads for data loading')
    parser.add_argument('--cpu', type=bool, default=False,
                        help='use cpu only')
    parser.add_argument('--n_GPUs', type=int, default=4,
                        help='number of GPUs')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    
    # Data specifications
    
    parser.add_argument('--data_train', type=str, default='/home/data/Task01_BrainTumour/imagesTr/',
                        help='train dataset directory')
    parser.add_argument('--data_test', type=str, default='/home/data/Task01_BrainTumour/imagesTs/',
                        help='test dataset directory')
    parser.add_argument('--scale', type=str, default='3',
                        help='super resolution scale')
    parser.add_argument('--n_channel', type=int, default=1,
                        help='number of color channels to use')
    parser.add_argument('--visual', type=str, default="ours",
                        help='number of color channels to use')
    parser.add_argument('--mode', type=str, default="ours",
                        help='mode to upscale')
    
    # Training specifications
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train the whole network')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='input batch size for training')
    parser.add_argument('--verbose',  action='store_true',
                        help='print training information. Default False')
    
    # Optimization specifications
    parser.add_argument('--rdn_lr', type=float, default=1e-4,
                        help='learning rate to train the whole network')
    parser.add_argument('--mtran_lr', type=float, default=1e-4,
                        help='learning rate to train the whole network')
    parser.add_argument('--ours_lr', type=float, default=1e-4,
                        help='learning rate to train the whole network')
    parser.add_argument('--lr_T2Net', type=float, default= 1e-4,
                        help='learning rate to train the whole network')
                        
  
    
    # Loss specifications
    parser.add_argument('--loss', type=str, default='1*L1',
                        help='loss function configuration')
                        
#masa
    parser.add_argument('--net_name', default='MASA', type=str, help='RefNet | Baseline')
    parser.add_argument('--sr_scale', default=4, type=int)
    parser.add_argument('--input_nc', default=1, type=int)
    parser.add_argument('--output_nc', default=1, type=int)
    parser.add_argument('--nf', default=64, type=int)
    parser.add_argument('--n_blks', default='4, 4, 4', type=str)
    parser.add_argument('--nf_ctt', default=32, type=int)
    parser.add_argument('--n_blks_ctt', default='2, 2, 2', type=str)
    parser.add_argument('--num_nbr', default=1, type=int)
    parser.add_argument('--n_blks_dec', default=10, type=int)
    parser.add_argument('--ref_level', default=1, type=int)
        
    args = parser.parse_args()
    return args
