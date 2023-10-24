import os
import sys
import logging
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from utils.utils import test_single_volume
from utils.dataset_ACDC import ACDCdataset, RandomGenerator
from lib.networks import PVT_GCASCADE, MERIT_GCASCADE
        
def inference(args, model, testloader, test_save_path=None):
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    with torch.no_grad():
        for i_batch, sampled_batch in tqdm(enumerate(testloader)):
            h, w = sampled_batch["image"].size()[2:]
            image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
            metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                          test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
            metric_list += np.array(metric_i)
            logging.info('idx %d case %s mean_dice %f mean_hd95 %f, mean_jacard %f mean_asd %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1], np.mean(metric_i, axis=0)[2], np.mean(metric_i, axis=0)[3]))
        metric_list = metric_list / len(testloader)
        for i in range(1, args.num_classes):
            logging.info('Mean class (%d) mean_dice %f mean_hd95 %f, mean_jacard %f mean_asd %f' % (i, metric_list[i-1][0], metric_list[i-1][1], metric_list[i-1][2], metric_list[i-1][3]))
        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        mean_jacard = np.mean(metric_list, axis=0)[2]
        mean_asd = np.mean(metric_list, axis=0)[3]
        logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f, mean_jacard : %f mean_asd : %f' % (performance, mean_hd95, mean_jacard, mean_asd))
        logging.info("Testing Finished!")
        return performance, mean_hd95, mean_jacard, mean_asd
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', default='PVT', help='Name of encoder: PVT or MERIT')
    parser.add_argument('--skip_aggregation', default='additive', help='Type of skip-aggregation: additive or concatenation')
    parser.add_argument("--batch_size", default=12, help="batch size")
    parser.add_argument("--lr", default=0.0001, help="learning rate")
    parser.add_argument("--max_epochs", default=400)
    parser.add_argument("--img_size", default=224)
    parser.add_argument("--save_path", default="./model_pth/ACDC")
    parser.add_argument("--n_gpu", default=1)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--list_dir", default="./data/ACDC/lists_ACDC")
    parser.add_argument("--root_dir", default="./data/ACDC/") 
    parser.add_argument("--volume_path", default="./data/ACDC/test")
    parser.add_argument("--z_spacing", default=10)
    parser.add_argument("--num_classes", default=4)
    parser.add_argument('--test_save_dir', default='./predictions', help='saving prediction as nii!')
    parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
    parser.add_argument('--seed', type=int,
                    default=2222, help='random seed')              
    args = parser.parse_args()

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

    args.is_pretrain = True
    args.exp = 'PVT_GCASCADE_MUTATION_w3_7_Run1_' + str(args.img_size)
    snapshot_path = "{}/{}/{}".format(args.save_path, args.exp, 'PVT_GCASCADE_MUTATION_w3_7_Run1')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.lr) if args.lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

    if args.encoder=='PVT':
        net = PVT_GCASCADE(n_class=args.num_classes, img_size=args.img_size, k=11, padding=5, conv='mr', gcb_act='gelu', skip_aggregation=args.skip_aggregation).cuda()
    elif args.encoder=='MERIT':
        net = MERIT_GCASCADE(n_class=args.num_classes, img_size_s1=(args.img_size,args.img_size), img_size_s2=(224,224), k=11, padding=5, conv='mr', gcb_act='gelu', skip_aggregation=args.skip_aggregation).cuda()
    else:
        print('Implementation not found for this encoder. Exiting!')
        sys.exit()

    snapshot = os.path.join(snapshot_path, 'best.pth')
    if not os.path.exists(snapshot): snapshot = snapshot.replace('best', 'epoch_'+str(args.max_epochs-1))
    net.load_state_dict(torch.load(snapshot))
    snapshot_name = snapshot_path.split('/')[-1]

    log_folder = 'test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    args.test_save_dir = os.path.join(snapshot_path, args.test_save_dir)
    test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
    os.makedirs(test_save_path, exist_ok=True)
    
    
    db_test =ACDCdataset(base_dir=args.volume_path,list_dir=args.list_dir, split="test")
    testloader = DataLoader(db_test, batch_size=1, shuffle=False)
    
    results = inference(args, net, testloader, test_save_path)


