import argparse
import logging
import os
import random
import numpy as np
import time

import torch
import torch.backends.cudnn as cudnn

from lib.networks import PVT_GCASCADE, MERIT_GCASCADE

from trainer import trainer_synapse
from torchsummaryX import summary
from ptflops import get_model_complexity_info


parser = argparse.ArgumentParser()
parser.add_argument('--encoder', type=str,
                    default='PVT', help='Name of encoder: PVT or MERIT')
parser.add_argument('--skip_aggregation', type=str,
                    default='additive', help='Type of skip-aggregation: additive or concatenation')
parser.add_argument('--root_path', type=str,
                    default='./data/synapse/train_npz', help='root dir for data')
parser.add_argument('--volume_path', type=str,
                    default='./data/synapse/test_vol_h5', help='root dir for validation volume data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=300, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=6, help='batch_size per gpu') #6
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.0001,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input') #256
parser.add_argument('--seed', type=int,
                    default=2222, help='random seed')

args = parser.parse_args()

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    
    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'root_path': args.root_path,
            'volume_path': args.volume_path,
            'list_dir': args.list_dir,
            'num_classes': args.num_classes,
            'z_spacing': 1,
        },
    }
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True
    args.exp = 'PVT_GCASCADE_MUTATION_w3_7_Run1_' + dataset_name + str(args.img_size)
    snapshot_path = "model_pth/{}/{}".format(args.exp, 'PVT_GCASCADE_MUTATION_w3_7_Run1')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path
    #current_time = time.strftime("%H%M%S")
    #print("The current time is", current_time)
    #snapshot_path = snapshot_path +'_t'+current_time

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    
    if args.encoder=='PVT':
        net = PVT_GCASCADE(n_class=args.num_classes, img_size=args.img_size, k=11, padding=5, conv='mr', gcb_act='gelu', skip_aggregation=args.skip_aggregation)
    elif args.encoder=='MERIT':
        net = MERIT_GCASCADE(n_class=args.num_classes, img_size_s1=(args.img_size,args.img_size), img_size_s2=(224,224), k=11, padding=5, conv='mr', gcb_act='gelu', skip_aggregation=args.skip_aggregation)
    else:
        print('Implementation not found for this encoder. Exiting!')
        sys.exit()

    print('Model %s created' % (args.encoder+'-GCASCADE: '))

    net = net.cuda()
   
    macs, params = get_model_complexity_info(net, (3, args.img_size, args.img_size), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    
    
    trainer = {'Synapse': trainer_synapse,}
    trainer[dataset_name](args, net, snapshot_path)
