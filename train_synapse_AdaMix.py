import argparse
import logging
import os
import random
import shutil
import sys
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from skimage.measure import label
from utils.displacement import ABD_R_BCP
from dataloaders.synapse import (BaseDataSets, TwoStreamBatchSampler, WeakStrongAugment)
from networks.net_factory import BCP_net
from utils import ramps, losses
from val_2D import test_single_volume
import numpy as np
from scipy.ndimage import binary_dilation

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/Synapse', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='AdaMix', help='experiment_name')
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--pre_iterations', type=int, default=10000, help='maximum epoch number to train')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--image_size', type=list,  default=[224, 224], help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=9, help='output channel of network')
# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=4, help='labeled_batch_size per gpu')
parser.add_argument('--labelnum', type=int, default=1, help='labeled data')
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
# costs
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float,  default='6.0', help='magnitude')
parser.add_argument('--s_param', type=int,  default=6, help='multinum of random masks')
# patch size
parser.add_argument('--patch_size', type=int, default=64, help='patch_size')
parser.add_argument('--h_size', type=int, default=4, help='h_size')
parser.add_argument('--w_size', type=int, default=4, help='w_size')
# top num
parser.add_argument('--top_num', type=int, default=4, help='top_num')
parser.add_argument('--m', type=float,help='(max + min)/m', default=3)
parser.add_argument('--lam', help='weight the loss function', default=1)


args = parser.parse_args()
dice_loss = losses.DiceLoss(n_classes=9)

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


def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])

def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])

def save_net_opt(net, optimizer, path):
    state = {
        'net':net.state_dict(),
        'opt':optimizer.state_dict(),
    }
    torch.save(state, str(path))

import torch
import numpy as np
from skimage.measure import label

def get_ACDC_2DLargestCC(segmentation):
    batch_list = []
    N = segmentation.shape[0]
    
    for i in range(N):
        class_array = np.zeros_like(segmentation[i].cpu().numpy())
        
        for c in range(1, 9):  # Class labels start from 1
            temp_seg = (segmentation[i] == c).cpu().numpy()
            labels = label(temp_seg)
            
            if labels.max() != 0:
                # Find the largest connected component
                largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
                class_array[largestCC] = c  # Assign class label
            
        batch_list.append(class_array)
    
    # Convert the batch list to a single NumPy array and then a PyTorch tensor
    batch_array = np.stack(batch_list, axis=0)
    return torch.from_numpy(batch_array).to(segmentation.device)


def get_ACDC_2DLargestCC(segmentation):
    batch_list = []
    N = segmentation.shape[0]
    
    for i in range(N):
        class_array = np.zeros_like(segmentation[i].cpu().numpy())
        
        for c in range(1, 9):  # Class labels start from 1
            temp_seg = (segmentation[i] == c).cpu().numpy()
            labels = label(temp_seg)
            
            if labels.max() != 0:
                # Find the largest connected component
                largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
                class_array[largestCC] = c  # Assign class label
            
        batch_list.append(class_array)
    
    # Convert the batch list to a single NumPy array and then a PyTorch tensor
    batch_array = np.stack(batch_list, axis=0)
    return torch.from_numpy(batch_array).to(segmentation.device)

def get_ACDC_masks(output, nms=0):
    probs = F.softmax(output, dim=1)
    _, probs = torch.max(probs, dim=1)
    if nms == 1:
        probs = get_ACDC_2DLargestCC(probs)      
    return probs

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 5* args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_model_ema(model, ema_model, alpha):
    model_state = model.state_dict()
    model_ema_state = ema_model.state_dict()
    new_dict = {}
    for key in model_state:
        new_dict[key] = alpha * model_ema_state[key] + (1 - alpha) * model_state[key]
    ema_model.load_state_dict(new_dict)

# def update_ema_variables(model, ema_model, alpha, global_step, args):
#     # adjust the momentum param
#     if global_step < args["consistency_rampup"]:
#         alpha = 0.0 
#     else:
#         alpha = min(1 - 1 / (global_step - args["consistency_rampup"] + 1), alpha)
    
#     # update weights
#     for ema_param, param in zip(ema_model.parameters(), model.parameters()):
#         ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
    
#     # update buffers
#     for buffer_train, buffer_eval in zip(model.buffers(), ema_model.buffers()):
#         buffer_eval.data = buffer_eval.data * alpha + buffer_train.data * (1 - alpha)


def generate_mask(img):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_x, patch_y = int(img_x*2/3), int(img_y*2/3)
    w = np.random.randint(0, img_x - patch_x)
    h = np.random.randint(0, img_y - patch_y)
    mask[w:w+patch_x, h:h+patch_y] = 0
    loss_mask[:, w:w+patch_x, h:h+patch_y] = 0
    return mask.long(), loss_mask.long()

def random_mask(img, shrink_param=3):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    x_split, y_split = int(img_x / shrink_param), int(img_y / shrink_param)
    patch_x, patch_y = int(img_x*2/(3*shrink_param)), int(img_y*2/(3*shrink_param))
    mask = torch.ones(img_x, img_y).cuda()
    for x_s in range(shrink_param):
        for y_s in range(shrink_param):
            w = np.random.randint(x_s*x_split, (x_s+1)*x_split-patch_x)
            h = np.random.randint(y_s*y_split, (y_s+1)*y_split-patch_y)
            mask[w:w+patch_x, h:h+patch_y] = 0
            loss_mask[:, w:w+patch_x, h:h+patch_y] = 0
    return mask.long(), loss_mask.long()

def contact_mask(img):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_y = int(img_y *4/9)
    h = np.random.randint(0, img_y-patch_y)
    mask[h:h+patch_y, :] = 0
    loss_mask[:, h:h+patch_y, :] = 0
    return mask.long(), loss_mask.long()


def mix_loss(output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    CE = nn.CrossEntropyLoss(reduction='none')
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    output_soft = F.softmax(output, dim=1)
    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    patch_mask = 1 - mask
    loss_dice = dice_loss(output_soft, img_l.unsqueeze(1), mask.unsqueeze(1)) * image_weight
    loss_dice += dice_loss(output_soft, patch_l.unsqueeze(1), patch_mask.unsqueeze(1)) * patch_weight
    loss_ce = image_weight * (CE(output, img_l) * mask).sum() / (mask.sum() + 1e-16) 
    loss_ce += patch_weight * (CE(output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)  # loss = loss_ce
    return loss_dice, loss_ce

def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"1": 32, "3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "70": 1312}
    elif "Synapse" in dataset:
        ref_dict = {"1": 23, "5": 111, "10": 222}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]

def pre_train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.pre_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs/2), int((args.batch_size-args.labeled_bs) / 2)
     
    model = BCP_net(in_chns=1, class_num=num_classes)
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(root=args.root_path,
                            mode="train",
                            num=None,
                            transform=transforms.Compose([WeakStrongAugment(args.image_size)]))
    db_val = BaseDataSets(root=args.root_path, mode="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path,args.labelnum)
    print("Total slices is: {}, labeled slices is:{}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start pre_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model.train()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_hd = 100
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]
            img_mask, loss_mask = generate_mask(img_a)
            gt_mixl = lab_a * img_mask + lab_b * (1 - img_mask)

            #-- original
            net_input = img_a * img_mask + img_b * (1 - img_mask)
            out_mixl = model(net_input)
            loss_dice, loss_ce = mix_loss(out_mixl, lab_a, lab_b, loss_mask, u_weight=1.0, unlab=True)

            loss = (loss_dice + loss_ce) / 2            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1

            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/mix_dice', loss_dice, iter_num)
            writer.add_scalar('info/mix_ce', loss_ce, iter_num)     

                
            if iter_num % 20 == 0:
                image = net_input[1, 0:1, :, :]
                writer.add_image('pre_train/Mixed_Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(out_mixl, dim=1), dim=1, keepdim=True)
                writer.add_image('pre_train/Mixed_Prediction', outputs[1, ...] * 50, iter_num)
                labs = gt_mixl[1, ...].unsqueeze(0) * 50
                writer.add_image('pre_train/Mixed_GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                dice_class = [0] * 8
                with torch.no_grad():
                    for sample in valloader:
                        img, mask =sample['image'].cuda(), sample['label'].cuda()
                        h, w = img.shape[-2:]
                        img = F.interpolate(
                            img,
                            (224, 224),
                            mode="bilinear",
                            align_corners=False,
                        )
                        img = img.permute(1, 0, 2, 3)
                        pred = model(img)
                        pred = F.interpolate(pred, (h, w), mode="bilinear", align_corners=False)
                        
                        pred = pred.argmax(dim=1).unsqueeze(0)
                        for cls in range(1, 9):
                            inter = ((pred == cls) * (mask == cls)).sum().item()
                            union = (pred == cls).sum().item() + (mask == cls).sum().item()
                            if union != 0:
                                dice_class[cls - 1] += 2.0 * inter / union

                dice_class = [dice * 100.0 / len(valloader) for dice in dice_class]
                mean_dice = sum(dice_class) / len(dice_class)
                    
                print(
                    "***** Evaluation ***** >>>> MeanDice: {:.2f}\n".format(mean_dice)
                )
                
                if mean_dice > best_performance:
                    best_performance = mean_dice
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model))
                    save_net_opt(model, optimizer, save_mode_path)
                    save_net_opt(model, optimizer, save_best_path)

                logging.info('iteration %d : mean_dice : %f' % (iter_num, mean_dice))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()

def self_train(args ,pre_snapshot_path, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    pre_trained_model = os.path.join(pre_snapshot_path,'{}_best_model.pth'.format(args.model))
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs/2), int((args.batch_size-args.labeled_bs) / 2)
     
    model_1 = BCP_net(in_chns=1, class_num=num_classes)
    model_2 = BCP_net(in_chns=1, class_num=num_classes)
    ema_model = BCP_net(in_chns=1, class_num=num_classes, ema=True)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(root=args.root_path,
                            mode="train",
                            num=None,
                            transform=transforms.Compose([WeakStrongAugment(args.image_size)]))
    db_val = BaseDataSets(root=args.root_path, mode="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path,args.labelnum)
    print("Train labeled {} samples".format(labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer1 = optim.SGD(model_1.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model_2.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    load_net(ema_model, pre_trained_model)
    load_net_opt(model_1, optimizer1, pre_trained_model)
    load_net_opt(model_2, optimizer2, pre_trained_model)
    logging.info("Loaded from {}".format(pre_trained_model))

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start self_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model_1.train()
    model_2.train()
    ema_model.train()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance1 = 0.0
    best_performance2 = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            volume_batch_strong, label_batch_strong = sampled_batch['image_strong'], sampled_batch['label_strong']
            volume_batch_strong, label_batch_strong = volume_batch_strong.cuda(), label_batch_strong.cuda()

            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]
            uimg_a, uimg_b = volume_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], volume_batch[args.labeled_bs + unlabeled_sub_bs:]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]

            img_a_s, img_b_s = volume_batch_strong[:labeled_sub_bs], volume_batch_strong[labeled_sub_bs:args.labeled_bs]
            uimg_a_s, uimg_b_s = volume_batch_strong[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], volume_batch_strong[args.labeled_bs + unlabeled_sub_bs:]
            lab_a_s, lab_b_s = label_batch_strong[:labeled_sub_bs], label_batch_strong[labeled_sub_bs:args.labeled_bs]

            with torch.no_grad():
                pre_a = ema_model(uimg_a)
                pre_b = ema_model(uimg_b)
                plab_a = get_ACDC_masks(pre_a, nms=1)  # plab_a.shape=[6, 224, 224]
                plab_b = get_ACDC_masks(pre_b, nms=1)
                
                pre_a_s = ema_model(uimg_a_s)
                pre_b_s = ema_model(uimg_b_s)
                plab_a_s = get_ACDC_masks(pre_a_s, nms=1)  # plab_a.shape=[6, 224, 224]
                plab_b_s = get_ACDC_masks(pre_b_s, nms=1)
                
                img_mask, loss_mask = generate_mask(img_a)
                # unl_label = ulab_a * img_mask + lab_a * (1 - img_mask)
                # l_label = lab_b * img_mask + ulab_b * (1 - img_mask)
            consistency_weight = get_current_consistency_weight(iter_num//150)

            origin_a = img_a.clone()
            origin_b = img_b.clone()
            origin_u_a = uimg_a.clone()
            origin_u_b = uimg_b.clone()
            
            origin_a_s = img_a_s.clone()
            origin_b_s = img_b_s.clone()
            origin_u_a_s = uimg_a_s.clone()
            origin_u_b_s = uimg_b_s.clone()
        
            net_input_unl_1 = uimg_a * img_mask + img_a * (1 - img_mask)
            net_input_l_1 = img_b * img_mask + uimg_b * (1 - img_mask)
            net_input_1 = torch.cat([net_input_unl_1, net_input_l_1], dim=0) 

            net_input_unl_2 = uimg_a_s * img_mask + img_a_s * (1 - img_mask)
            net_input_l_2 = img_b_s * img_mask + uimg_b_s * (1 - img_mask)
            net_input_2 = torch.cat([net_input_unl_2, net_input_l_2], dim=0)

            # Model1 Loss: torch.Size([6, 1, 256, 256])  
            out_unl_1 = model_1(net_input_unl_1)
            out_l_1 = model_1(net_input_l_1)
            out_1 = torch.cat([out_unl_1, out_l_1], dim=0)
            out_soft_1 = torch.softmax(out_1, dim=1)
            
            out_max_1 = torch.max(out_soft_1.detach(), dim=1)[0]
            out_pseudo_1 = torch.argmax(out_soft_1.detach(), dim=1, keepdim=False) 
            
            unl_dice_1, unl_ce_1 = mix_loss(out_unl_1, plab_a, lab_a, loss_mask, u_weight=args.u_weight, unlab=True)
            l_dice_1, l_ce_1 = mix_loss(out_l_1, lab_b, plab_b, loss_mask, u_weight=args.u_weight)
            loss_ce_1 = unl_ce_1 + l_ce_1
            loss_dice_1 = unl_dice_1 + l_dice_1

            # Model2 Loss
            out_unl_2 = model_2(net_input_unl_2)
            out_l_2 = model_2(net_input_l_2)
            out_2 = torch.cat([out_unl_2, out_l_2], dim=0)
            out_soft_2 = torch.softmax(out_2, dim=1)
            out_max_2 = torch.max(out_soft_2.detach(), dim=1)[0]
            out_pseudo_2 = torch.argmax(out_soft_2.detach(), dim=1, keepdim=False) 
            
            unl_dice_2, unl_ce_2 = mix_loss(out_unl_2, plab_a_s, lab_a_s, loss_mask, u_weight=args.u_weight, unlab=True)
            l_dice_2, l_ce_2 = mix_loss(out_l_2, lab_b_s, plab_b_s, loss_mask, u_weight=args.u_weight)
            loss_ce_2 = unl_ce_2 + l_ce_2
            loss_dice_2 = unl_dice_2 + l_dice_2

            # Model1 & Model2 Cross Pseudo Supervision
            pseudo_supervision1 = dice_loss(out_soft_1, out_pseudo_2.unsqueeze(1))  
            pseudo_supervision2 = dice_loss(out_soft_2, out_pseudo_1.unsqueeze(1))  
            
            # exchange the area based on the size      
            model_1_output_u_a = model_1(origin_u_a)
            model_1_output_u_b = model_1(origin_u_b)
            model_2_output_u_a = model_2(origin_u_a_s)
            model_2_output_u_b = model_2(origin_u_b_s)
            
            model_1_output_u_a_soft = torch.softmax(model_1_output_u_a, dim=1)
            model_1_output_u_b_soft = torch.softmax(model_1_output_u_b, dim=1)
            model_2_output_u_a_soft = torch.softmax(model_2_output_u_a, dim=1)
            model_2_output_u_b_soft = torch.softmax(model_2_output_u_b, dim=1)

            max_a, _ =  torch.max(model_1_output_u_a_soft, dim=1)
            max_b, _ =  torch.max(model_1_output_u_b_soft, dim=1)
            max_c, _ =  torch.max(model_2_output_u_a_soft, dim=1)
            max_d, _ =  torch.max(model_2_output_u_b_soft, dim=1)

            # ([6, 256, 256]) 
            # compute theta
            max_a_mean = (max_a.view(max_a.size(0), -1).max(dim=1).values + max_a.view(max_a.size(0), -1).min(dim=1).values) / args.m
            max_b_mean = (max_b.view(max_b.size(0), -1).max(dim=1).values + max_b.view(max_b.size(0), -1).min(dim=1).values) /  args.m
            max_c_mean = (max_c.view(max_c.size(0), -1).max(dim=1).values + max_c.view(max_c.size(0), -1).min(dim=1).values) /  args.m
            max_d_mean = (max_d.view(max_d.size(0), -1).max(dim=1).values + max_d.view(max_d.size(0), -1).min(dim=1).values) /  args.m
            
            # print(max_a.shape, max_a_mean.shape)
            mask_model_1_a = (max_a < max_a_mean.unsqueeze(1).unsqueeze(2).expand(2, 224, 224)).int().unsqueeze(1)
            mask_model_1_b = (max_b < max_b_mean.unsqueeze(1).unsqueeze(2).expand(2, 224, 224)).int().unsqueeze(1)
            mask_model_2_a = (max_c < max_c_mean.unsqueeze(1).unsqueeze(2).expand(2, 224, 224)).int().unsqueeze(1)
            mask_model_2_b = (max_d < max_d_mean.unsqueeze(1).unsqueeze(2).expand(2, 224, 224)).int().unsqueeze(1)
            
            mask_model_1_a = get_exchange_mask(mask_model_1_a.squeeze(1)).unsqueeze(1)
            mask_model_1_b = get_exchange_mask(mask_model_1_b.squeeze(1)).unsqueeze(1)
            mask_model_2_a = get_exchange_mask(mask_model_2_a.squeeze(1)).unsqueeze(1)
            mask_model_2_b = get_exchange_mask(mask_model_2_b.squeeze(1)).unsqueeze(1)
            
            net_input_unlab_1 = origin_u_a * mask_model_1_a + origin_a * (1 - mask_model_1_a)
            net_input_lab_1 = origin_b * mask_model_1_b + origin_u_b * (1- mask_model_1_b)
            

            net_input_unlab_2 = origin_u_a_s * mask_model_2_a + origin_a_s * (1 - mask_model_2_a)
            net_input_lab_2 = origin_b_s * mask_model_2_b + origin_u_b_s * (1- mask_model_2_b)
            
            # mask: torch.Size([6, 6, 256, 256]) 
            # plab_a: torch.Size([6, 256, 256])
            # mask_model_1_a: torch.Size([6, 1, 256, 256])
            # mask: torch.Size([6, 6, 256, 256])
            mask = plab_a.unsqueeze(1) * mask_model_1_a + lab_a.unsqueeze(1) * (1 - mask_model_1_a)

            
            net_first_output_1 = model_1(net_input_unlab_1)
            net_secode_output_1 = model_1(net_input_lab_1)
            net_first_output_2 = model_2(net_input_unlab_2)
            net_second_output_2 = model_2(net_input_lab_2)
            
            # ## 
            # writer.add_image('net_first_output_1', net_first_output_1[0], iter_num)
            # writer.add_image('net_first_output_2', net_first_output_2[0], iter_num)

            # cps
            out_soft_3 = torch.softmax(torch.cat([net_first_output_1, net_secode_output_1], dim=0), dim=1)
            out_soft_4 = torch.softmax(torch.cat([net_first_output_2, net_second_output_2], dim=0), dim=1)
            out_pseudo_3 = torch.argmax(out_soft_3.detach(), dim=1, keepdim=False) 
            out_pseudo_4 = torch.argmax(out_soft_4.detach(), dim=1, keepdim=False)
            
            loss_dice_add_1_a, loss_ce_add_1_a = mix_loss(net_first_output_1, plab_a, lab_a, mask_model_1_a, u_weight=args.u_weight, unlab=True)
            loss_dice_add_1_b, loss_ce_add_1_b = mix_loss(net_secode_output_1, lab_b, plab_b, mask_model_1_b, u_weight=args.u_weight)
            
            loss_dice_add_2_a, loss_ce_add_2_a = mix_loss(net_first_output_2, plab_a_s, lab_a_s, mask_model_2_a, u_weight=args.u_weight, unlab=True)
            loss_dice_add_2_b, loss_ce_add_2_b = mix_loss(net_second_output_2, lab_b_s, plab_b_s, mask_model_2_b, u_weight=args.u_weight)
            
            loss_ce_mix_1 =  loss_ce_add_1_a + loss_ce_add_1_b
            loss_dice_mix_1 = loss_dice_add_1_a + loss_dice_add_1_b
            
            loss_ce_mix_2 =  loss_ce_add_2_a + loss_ce_add_2_b
            loss_dice_mix_2 = loss_dice_add_2_a + loss_dice_add_2_b
            
            pseudo_supervision3 = dice_loss(out_soft_3, out_pseudo_4.unsqueeze(1))    
            pseudo_supervision4 = dice_loss(out_soft_4, out_pseudo_3.unsqueeze(1))
            
            loss_1 =  args.lam * (loss_ce_mix_1 + loss_dice_mix_1) / 4  + (loss_dice_1 + loss_ce_1) / 4 + (pseudo_supervision1 + pseudo_supervision3)/ 4
            loss_2 =  args.lam * (loss_ce_mix_2 + loss_dice_mix_2) / 4 + (loss_dice_2 + loss_ce_2) / 4  +  (pseudo_supervision2 + pseudo_supervision4) / 4
            loss = loss_1 + loss_2

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss.backward()
            optimizer1.step()
            optimizer2.step()

            iter_num += 1
            update_model_ema(model_1, ema_model, 0.99)

            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/model1_loss', loss_1, iter_num)
            writer.add_scalar('info/model2_loss', loss_2, iter_num)
            writer.add_scalar('info/model1/mix_dice', loss_dice_1, iter_num)
            writer.add_scalar('info/model1/mix_ce', loss_ce_1, iter_num)
            writer.add_scalar('info/model2/mix_dice', loss_dice_2, iter_num)
            writer.add_scalar('info/model2/mix_ce', loss_ce_2, iter_num)
            writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)     


            if iter_num > 0 and iter_num % 200 == 0:
                model_1.eval()
                dice_class_1 = [0] * 8
                with torch.no_grad():
                    for sample in valloader:
                        img, mask =sample['image'].cuda(), sample['label'].cuda()
                        h, w = img.shape[-2:]
                        img = F.interpolate(
                            img,
                            (224, 224),
                            mode="bilinear",
                            align_corners=False,
                        )
                        img = img.permute(1, 0, 2, 3)
                        pred = model_1(img)
                        pred = F.interpolate(pred, (h, w), mode="bilinear", align_corners=False)
                        
                        pred = pred.argmax(dim=1).unsqueeze(0)
                        for cls in range(1, 9):
                            inter = ((pred == cls) * (mask == cls)).sum().item()
                            union = (pred == cls).sum().item() + (mask == cls).sum().item()
                            if union != 0:
                                dice_class_1[cls - 1] += 2.0 * inter / union

                dice_class_1 = [dice * 100.0 / len(valloader) for dice in dice_class_1]
                mean_dice_1 = sum(dice_class_1) / len(dice_class_1)
                    
                print("***** Evaluation ***** >>>> MeanDice: {:.2f}\n".format(mean_dice_1))
                
                if mean_dice_1 > best_performance1:
                    best_performance1 = mean_dice_1
                    save_mode_path = os.path.join(snapshot_path, 'model1_iter_{}_dice_{}.pth'.format(iter_num, round(best_performance1, 4)))
                    save_best_path = os.path.join(snapshot_path,'model1_{}_best_model.pth'.format(args.model))
                    save_net_opt(model_1, optimizer1, save_mode_path)
                    save_net_opt(model_1, optimizer1, save_best_path)

                logging.info('iteration %d : mean_dice_1 : %f' % (iter_num, mean_dice_1))
                model_1.train()

                model_2.eval()
                dice_class_2 = [0] * 8
                with torch.no_grad():
                    for sample in valloader:
                        img, mask =sample['image'].cuda(), sample['label'].cuda()
                        h, w = img.shape[-2:]
                        img = F.interpolate(
                            img,
                            (224, 224),
                            mode="bilinear",
                            align_corners=False,
                        )
                        img = img.permute(1, 0, 2, 3)
                        pred = model_2(img)
                        pred = F.interpolate(pred, (h, w), mode="bilinear", align_corners=False)
                        
                        pred = pred.argmax(dim=1).unsqueeze(0)
                        for cls in range(1, 9):
                            inter = ((pred == cls) * (mask == cls)).sum().item()
                            union = (pred == cls).sum().item() + (mask == cls).sum().item()
                            if union != 0:
                                dice_class_2[cls - 1] += 2.0 * inter / union

                dice_class_2 = [dice * 100.0 / len(valloader) for dice in dice_class_2]
                mean_dice_2 = sum(dice_class_2) / len(dice_class_2)
                    
                print(
                    "***** Evaluation ***** >>>> MeanDice: {:.2f}\n".format(mean_dice_2)
                )
                
                if mean_dice_2 > best_performance2:
                    best_performance2 = mean_dice_2
                    save_mode_path = os.path.join(snapshot_path, 'model2_iter_{}_dice_{}.pth'.format(iter_num, round(best_performance2, 4)))
                    save_best_path = os.path.join(snapshot_path,'model2_{}_best_model.pth'.format(args.model))
                    save_net_opt(model_2, optimizer2, save_mode_path)
                    save_net_opt(model_2, optimizer2, save_best_path)

                logging.info('iteration %d : mean_dice_2 : %f' % (iter_num, mean_dice_2))
                model_2.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    else:
        cudnn.benchmark = True
        cudnn.deterministic = False
        
    # -- path to save models
    pre_snapshot_path = "./model/AdaMix/synapse_AdaMix_{}_{}_m_{}_lam_{}_labeled_exange_fea_out_final_have_ori_cps_loss_RE/pre_train".format(args.exp, args.labelnum, args.m, args.lam)
    self_snapshot_path = "./model/AdaMix/synapse_AdaMix_{}_{}_m_{}_lam_{}_labeled_exange_fea_out_final_have_ori_cps_loss_RE/self_train".format(args.exp, args.labelnum,  args.m, args.lam)
    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
    shutil.copy('../code/train_synapse_AdaMix.py', self_snapshot_path)

    # Pre_train
    logging.basicConfig(filename=pre_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    pre_train(args, pre_snapshot_path)
    

    # Self_train
    logging.basicConfig(filename=self_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args, pre_snapshot_path, self_snapshot_path)