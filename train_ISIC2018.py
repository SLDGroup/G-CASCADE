import os
import numpy as np
import argparse
from datetime import datetime
import logging
import time

import torch
import torch.nn.functional as F
from torch.autograd import Variable

import matplotlib.pyplot as plt

from lib.networks import PVT_GCASCADE_SA_Grapher1st_UpConv2, PVT_GCASCADE_SA_Cat_Grapher1st_UpConv2, DoubleMaxViT_S_GCASCADE_SA_Cat_Grapher1st_UpConv2_C, DoubleMaxViT_S_GCASCADE_SA_Grapher1st_UpConv2_C

from utils.dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter
from utils.utils import powerset

from ptflops import get_model_complexity_info
from torchsummaryX import summary


l = [0, 1, 2, 3]
ss = [x for x in powerset(l)]
#ss = [[0],[1],[2],[3]]
print(ss)
        
def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
 
    return (wbce + wiou).mean()


def test(model, path, dataset):

    data_path = os.path.join(path, dataset)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    model.eval()
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, opt.img_size)
    DSC = 0.0
    for i in range(num1):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res1, res2, res3, res4 = model(image) # forward
        
        
        res = F.upsample(res1 + res2 + res3 + res4, size=gt.shape, mode='bilinear', align_corners=False) # additive aggregation and upsampling
        res = res.sigmoid().data.cpu().numpy().squeeze() # apply sigmoid activation for binary segmentation
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        
        # eval Dice
        input = res >= 0.5
        target = np.array(gt >= 0.5)
        N = gt.shape
        smooth = 1
        input_flat = np.reshape(input, (-1))
        target_flat = np.reshape(target, (-1))
        intersection = (input_flat * target_flat)
        dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
        dice = '{:.4f}'.format(dice)
        dice = float(dice)
        DSC = DSC + dice

    return DSC / num1, num1

def train(train_loader, model, optimizer, epoch, test_path, model_name = 'PVT-GCASCADE'):
    model.train()
    global best
    global total_train_time
    time_before_epoch_start = time.time()
    size_rates = [1] 
    loss_record = AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
        
            # ---- rescale ----
            trainsize = int(round(opt.img_size * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            
            # ---- forward ----
            P = model(images)
                       
            loss = 0.0
          
            for s in ss:
                iout = 0.0
                if(s==[]):
                    continue
                for idx in range(len(s)):
                    iout += P[s[idx]]
                loss += structure_loss(iout, gts)

            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_record.update(loss.data, opt.batchsize)
                
        # ---- train visualization ----
        if i % 50 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' loss: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record.show()))
            
    time_after_epoch_end = time.time()
    total_train_time += (time_after_epoch_end - time_before_epoch_start)
    print('total train time till current epoch: '+ str(total_train_time))
    logging.info('total train time till current epoch: '+ str(total_train_time))
    # save model 
    save_path = (opt.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), save_path + '' + model_name + '-last.pth')
    # choose the best model

    global dict_plot
   
    if (epoch + 1) % 1 == 0:
    	total_dice = 0
    	total_images = 0
    	for dataset in ['valid']:
    	    dataset_dice, n_images = test(model, test_path, dataset)
    	    total_dice += (n_images*dataset_dice)
    	    total_images += n_images
    	    logging.info('epoch: {}, dataset: {}, dice: {}'.format(epoch, dataset, dataset_dice))
    	    print(dataset, ': ', dataset_dice)
    	    dict_plot[dataset].append(dataset_dice)
    	dataset_test_dice, n_images = test(model, test_path, 'test')
    	meandice = total_dice/total_images
    	dict_plot['valid'].append(meandice)
    	dict_plot['test'].append(dataset_test_dice)
    	print('Test dice score: {}'.format(dataset_test_dice))
    	logging.info('Test dice score: {}'.format(dataset_test_dice))
    	if meandice > best:
            print('##################### Dice score improved from {} to {}'.format(best, meandice))
            logging.info('##################### Dice score improved from {} to {}'.format(best, meandice))
            best = meandice
            torch.save(model.state_dict(), save_path + '' + model_name + '-best.pth')
    
if __name__ == '__main__':
    dict_plot = {'valid':[],'test':[]}
    name = ['valid','test']
    ##################model_name#############################
    model_name = 'ISIC2018_811_PVT_GCASCADE_img_size384bs4_Run1'
    
    #current_time = time.strftime("%H%M%S")
    #print("The current time is", current_time)
    #model_name = model_name +'_t'+current_time
    
    ###############################################
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--encoder', type=str,
                        default='PVT', help='Name of encoder: PVT or MERIT')
    
    parser.add_argument('--skip_aggregation', type=str,
                        default='additive', help='Type of skip-aggregation: additive or concatenation')
                        
    parser.add_argument('--epoch', type=int,
                        default=200, help='epoch number')

    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')

    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='choosing optimizer AdamW or SGD')

    parser.add_argument('--augmentation',
                        default=False, help='choose to do random flip rotation')

    parser.add_argument('--batchsize', type=int,
                        default=4, help='training batch size')

    parser.add_argument('--img_size', type=int,
                        default=384, help='training dataset size')

    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')

    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')

    parser.add_argument('--decay_epoch', type=int,
                        default=200, help='every n epochs decay learning rate')

    parser.add_argument('--train_path', type=str,
                        default='./data/ISIC2018/train/',
                        help='path to train dataset')

    parser.add_argument('--test_path', type=str,
                        default='./data/ISIC2018/',
                        help='path to testing Kvasir dataset')

    parser.add_argument('--train_save', type=str,
                        default='./model_pth/'+model_name+'/')

    opt = parser.parse_args()
    
    
    
    logging.basicConfig(filename='train_log_'+model_name+'.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    # ---- build models ----
    #torch.cuda.set_device(0)  # set your gpu device
    if opt.encoder=='PVT':
        model = PVT_GCASCADE(n_class=1, img_size=opt.img_size, k=11, padding=5, conv='mr', act='gelu', skip_aggregation=opt.skip_aggregation)
    elif opt.encoder=='MERIT':
        model = MERIT_GCASCADE(n_class=1, img_size_s1=(256,256), img_size_s2=(224,224), k=11, padding=5, conv='mr', act='gelu', skip_aggregation=opt.skip_aggregation)
    else:
        print('Implementation not found for this encoder')

    print('Model %s created' % (encoder+'-GCASCADE: '))
    
    model.cuda()
    macs, params = get_model_complexity_info(model, (3, opt.img_size, opt.img_size), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
	
    best = 0

    params = model.parameters()
    
    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)

    print(optimizer)
    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.img_size, shuffle=True,
                              augmentation=opt.augmentation)
    total_step = len(train_loader)

    print("#" * 20, "Start Training", "#" * 20)
    total_train_time = 0

    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch, opt.test_path, model_name = model_name)
    print('avg train time: '+ str(total_train_time/(opt.epoch-1)))
    logging.info('avg train time: '+ str(total_train_time/(opt.epoch-1)))
