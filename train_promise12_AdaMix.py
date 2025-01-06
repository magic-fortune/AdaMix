import argparse
import logging
import os
import random
import shutil
import sys
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
import torch, gc
import torch.nn.functional as F
from medpy import metric
from PIL import Image
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataloaders.dataset import (BaseDataSets, RandomGenerator, TwoStreamBatchSampler, WeakStrongAugment)
from dataloaders.promise12 import Promise12
from networks.net_factory import net_factory
from networks.vision_transformer import SwinUnet as ViT_seg
from networks.config import get_config
from utils import losses, ramps
from val_2D import test_single_volume_promise
from utils.displacement import ABD_I, ABD_R


gc.collect()
torch.cuda.empty_cache()
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/promise12', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='train_PROMISE12', help='experiment_name')
parser.add_argument('--model_1', type=str,
                    default='unet', help='model1_name')
parser.add_argument('--model_2', type=str,
                    default='swin_unet', help='model2_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--image_size', type=list, default=[224, 224],
                    help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--num_classes', type=int, default=2,
                    help='output channel of network')
parser.add_argument('--cfg', type=str,
                    default="../code/configs/swin_tiny_patch4_window7_224_lite.yaml",
                    help='path to config file', )
parser.add_argument('--u_weight', help='the radio of unlabel', 
                    default=0.5)
parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs. ",
                    default=None, nargs='+', )
parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, ''full: cache all data, ''part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')
# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=8,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=3,
                    help='labeled data')
# costs
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
# patch size
parser.add_argument('--patch_size', type=int, default=56, help='patch_size')
parser.add_argument('--h_size', type=int, default=4, help='h_size')
parser.add_argument('--w_size', type=int, default=4, help='w_size')
# top num
parser.add_argument('--top_num', type=int, default=4, help='top_num')
args = parser.parse_args()  
config = get_config(args)

dice_loss = losses.DiceLoss(n_classes=2)
def mix_loss(output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    CE = nn.CrossEntropyLoss(reduction='none')
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    output_soft = F.softmax(output, dim=1)
    # print(output_soft.shape)
    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    patch_mask = 1 - mask
    loss_dice = dice_loss(output_soft, img_l.unsqueeze(1), mask.unsqueeze(1)) * image_weight
    loss_dice += dice_loss(output_soft, patch_l.unsqueeze(1), patch_mask.unsqueeze(1)) * patch_weight
    loss_ce = image_weight * (CE(output, img_l) * mask).sum() / (mask.sum() + 1e-16) 
    loss_ce += patch_weight * (CE(output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)  # loss = loss_ce
    return loss_dice, loss_ce


def get_exchange_mask(mask):
    # [6, 256, 256]
    d, h, w = mask.shape
    indices = torch.nonzero(mask, as_tuple=True) 
    try:
        min_h, min_w = indices[1].min().item(), indices[2].min().item()
        max_h, max_w = indices[1].max().item(), indices[2].max().item()
    except:
        min_h, min_w = (1//3) * h, (1//3) *  w
        max_h, max_w = (2//3) * h, (2//3) *  w
    
    exchange_mask = torch.ones_like(mask)
    exchange_mask[:, min_h - 5:max_h+6, min_w - 5:max_w+6] = 0
    return exchange_mask

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch,args.consistency_rampup)  # args.consistency=0.1 # args.consistency_rampup=200

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    return dice

def train(args, snapshot_path):
    base_lr = args.base_lr  
    num_classes = args.num_classes  
    batch_size = args.batch_size  
    max_iterations = args.max_iterations 
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu 

    def create_model(ema=False):
        # Network definition
        model = net_factory(net_type=args.model_1, in_chns=1, class_num=num_classes)  
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model1 = create_model() 
    model2 = ViT_seg(config, img_size=args.image_size, num_classes=args.num_classes).cuda()  
    model2.load_from(config)
    # model2 = create_model()
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = Promise12("../data/promise12", mode='train', out_size=224)
    db_val = Promise12("../data/promise12", mode='val', out_size=224)

    total_slices = len(db_train)
    # Total silices is: 1012, labeled slices is: 202                                                                                                                                                         
    labeled_slice = 101  # args.labeled_num=7 : 202 || label_num=3:101
    print("Total silices is: {}, labeled slices is: {}".format(total_slices, labeled_slice))

    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs)  # args.labeled_bs=8
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    loader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=0)

    model1.train()
    model2.train()

    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr,momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1  

    best_performance1 = 0.0
    best_performance2 = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)  

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['mask']
            volume_batch_strong, label_batch_strong = sampled_batch['image_strong'], sampled_batch['mask_strong']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            volume_batch_strong, label_batch_strong = volume_batch_strong.cuda(), label_batch_strong.cuda()

            origin_a, lab= volume_batch[:args.labeled_bs].clone(), label_batch[:args.labeled_bs].clone()
            origin_u_a  = volume_batch[args.labeled_bs:].clone()
            origin_a_s, lab_s = volume_batch_strong[args.labeled_bs:].clone(), label_batch_strong[args.labeled_bs:].clone()
            origin_u_a_s = volume_batch_strong[args.labeled_bs:]
            
            outputs1 = model1(volume_batch)  
            outputs1_unlabel = outputs1[args.labeled_bs:]
            outputs_soft1 = torch.softmax(outputs1, dim=1)
            outputs1_max = torch.max(outputs_soft1.detach(), dim=1)[0]
            pseudo_outputs1 = torch.argmax(outputs_soft1[args.labeled_bs:].detach(), dim=1, keepdim=False)  

            outputs2 = model2(volume_batch_strong)  
            outputs2_unlabel = outputs2[args.labeled_bs:]
            outputs_soft2 = torch.softmax(outputs2, dim=1)
            outputs2_max = torch.max(outputs_soft2.detach(), dim=1)[0]
            pseudo_outputs2 = torch.argmax(outputs_soft2[args.labeled_bs:].detach(), dim=1, keepdim=False)  
            
            model_1_output_u_a = model1(origin_u_a)
            model_2_output_u_a = model1(origin_u_a_s)
            
            model_1_output_u_a_soft = torch.softmax(model_1_output_u_a, dim=1)
            model_2_output_u_a_soft = torch.softmax(model_2_output_u_a, dim=1)

            # 16, 224, 224
            max_a, _ =  torch.max(model_1_output_u_a_soft, dim=1)
            max_c, _ =  torch.max(model_2_output_u_a_soft, dim=1)
            # print(max_a.shape)

            # ([6, 256, 256]) 
            # compute theta
            
            max_a_mean = (max_a.view(max_a.size(0), -1).max(dim=1).values + max_a.view(max_a.size(0), -1).min(dim=1).values) / 2  
            max_c_mean = (max_c.view(max_c.size(0), -1).max(dim=1).values + max_c.view(max_c.size(0), -1).min(dim=1).values) / 2 
            
            mask_model_1_a = (max_a < max_a_mean.unsqueeze(1).unsqueeze(2).expand(8, 224, 224)).int().unsqueeze(1)
            mask_model_2_a = (max_c < max_c_mean.unsqueeze(1).unsqueeze(2).expand(8, 224, 224)).int().unsqueeze(1)
            
            mask_model_1_a = get_exchange_mask(mask_model_1_a.squeeze(1)).unsqueeze(1)
            mask_model_2_a = get_exchange_mask(mask_model_2_a.squeeze(1)).unsqueeze(1)

            net_input_unlab_1 = origin_u_a * mask_model_1_a + origin_a * (1 - mask_model_1_a)
            net_input_unlab_2 = origin_u_a_s * mask_model_2_a + origin_a_s * (1 - mask_model_2_a)
            
            # mask: torch.Size([6, 6, 256, 256]) 
            # plab_a: torch.Size([6, 256, 256])
            # mask_model_1_a: torch.Size([6, 1, 256, 256])
            # mask: torch.Size([6, 6, 256, 256])
            
            net_first_output_1 = model1(net_input_unlab_1)
            net_first_output_2 = model1(net_input_unlab_2)
            # torch.Size([8, 2, 224, 224]) torch.Size([8, 224, 224]) torch.Size([8, 224, 224])
            # cps
            out_soft_3 = torch.softmax(net_first_output_1, dim=1)
            out_soft_4 = torch.softmax(net_first_output_2, dim=1)
            out_pseudo_3 = torch.argmax(out_soft_3.detach(), dim=1, keepdim=False) 
            out_pseudo_4 = torch.argmax(out_soft_4.detach(), dim=1, keepdim=False)
            
            loss_dice_add_1_a, loss_ce_add_1_a = mix_loss(net_first_output_1, pseudo_outputs1, lab, mask_model_1_a, u_weight=args.u_weight, unlab=True)
            loss_dice_add_2_a, loss_ce_add_2_a = mix_loss(net_first_output_2, pseudo_outputs2, lab_s, mask_model_2_a, u_weight=args.u_weight, unlab=True)
            
            loss_ce_mix_1 =  loss_ce_add_1_a 
            loss_dice_mix_1 = loss_dice_add_1_a 
            loss_ce_mix_2 =  loss_ce_add_2_a 
            loss_dice_mix_2 = loss_dice_add_2_a

            # origin loss
            loss1 = 0.5 * (ce_loss(outputs1[:args.labeled_bs], label_batch[:args.labeled_bs].long()) + dice_loss(outputs_soft1[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))
            loss2 = 0.5 * (ce_loss(outputs2[:args.labeled_bs], label_batch_strong[:args.labeled_bs].long()) + dice_loss(outputs_soft2[:args.labeled_bs], label_batch_strong[:args.labeled_bs].unsqueeze(1)))
            pseudo_supervision1 = dice_loss(outputs_soft1[args.labeled_bs:], pseudo_outputs2.unsqueeze(1))
            pseudo_supervision2 = dice_loss(outputs_soft2[args.labeled_bs:], pseudo_outputs1.unsqueeze(1))
        
            loss3 = 0.5 * (loss_ce_mix_1 + loss_dice_mix_1)
            loss4 = 0.5 * (loss_ce_mix_2 + loss_dice_mix_2)
            pseudo_supervision3 = dice_loss(out_soft_3, out_pseudo_4.unsqueeze(1))    
            pseudo_supervision4 = dice_loss(out_soft_4, out_pseudo_3.unsqueeze(1))
            # Total Loss
            consistency_weight = get_current_consistency_weight(iter_num // 150)
            model1_loss = loss1 + loss3 + consistency_weight * (pseudo_supervision1 + pseudo_supervision3)
            model2_loss = loss2 + loss4 + consistency_weight * (pseudo_supervision2 + pseudo_supervision4)
            loss = model1_loss + model2_loss

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss.backward()
            optimizer1.step()
            optimizer2.step()

            iter_num = iter_num + 1
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9

            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr_

            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('consistency_weight/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('loss/model1_loss', model1_loss, iter_num)
            writer.add_scalar('loss/model2_loss', model2_loss, iter_num)

            logging.info('iteration %d : model1 loss : %f model2 loss : %f' % (iter_num, model1_loss.item(), model2_loss.item()))
            if iter_num > 0 and iter_num % 200 == 0:
                model1.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(loader):
                    metric_i = test_single_volume_promise(sampled_batch["image"], sampled_batch["mask"], model1, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                writer.add_scalar('info/model1_val_{}_dice'.format(1),metric_list[0], iter_num)
                performance1 = np.mean(metric_list, axis=0)
                if performance1 > best_performance1:
                    best_performance1 = performance1
                    save_mode_path = os.path.join(snapshot_path,'model1_iter_{}_dice_{}.pth'.format(iter_num, round(best_performance1, 4)))
                    save_best = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model_1))
                    torch.save(model1.state_dict(), save_mode_path)
                    torch.save(model1.state_dict(), save_best)

                logging.info('iteration %d : model1_mean_dice : %f' % (iter_num, performance1))
                model1.train()

                model2.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(loader):
                    metric_i = test_single_volume_promise(sampled_batch["image"], sampled_batch["mask"], model2, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                writer.add_scalar('info/model2_val_{}_dice'.format(1),metric_list[0], iter_num)
                performance2 = np.mean(metric_list, axis=0)
                if performance2 > best_performance2:
                    best_performance2 = performance2
                    save_mode_path = os.path.join(snapshot_path,'model2_iter_{}_dice_{}.pth'.format(iter_num, round(best_performance2, 4)))
                    save_best = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model_2))
                    torch.save(model2.state_dict(), save_mode_path)
                    torch.save(model2.state_dict(), save_best)
                logging.info('iteration %d : model2_mean_dice : %f' % (iter_num, performance2))
                model2.train()

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


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

    snapshot_path = "./model/Cross_Teaching/PROMISE12_{}_{}_vit".format(args.exp, args.labeled_num)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    shutil.copy('../code/train_promise12_fix.py', snapshot_path)

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
