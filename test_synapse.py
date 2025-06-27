import argparse
import os
import shutil
import h5py
import numpy as np
import SimpleITK as sitk
from networks.net_factory import BCP_net
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
from networks.config import get_config
from networks.net_factory import net_factory
from networks.vision_transformer import SwinUnet as ViT_seg

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/Synapse/', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='AdaMix', help='experiment_name')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--model_1', type=str,
                    default='unet', help='model_name')
parser.add_argument('--model_2', type=str,
                    default='swin_unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=9,
                    help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=7,
                    help='labeled data')
parser.add_argument('--image_size', type=list, default=[224, 224],
                    help='patch size of network input')
parser.add_argument('--cfg', type=str,
                    default="./configs/swin_tiny_patch4_window7_224_lite.yaml",
                    help='path to config file', )
parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs. ",
                    default=None, nargs='+', )
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu')
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
parser.add_argument('--m', help='(max + min)/m', default=1.5)
parser.add_argument('--lam', help='weight the loss function', default=1.5)

args = parser.parse_args() 
config = get_config(args)

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if np.sum(pred) == 0:
        pred[0, 0] = 1
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    asd = metric.binary.asd(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    return dice, jc,asd, hd95



def test_single_volume(case, net, FLAGS):
    h5f = h5py.File(FLAGS.root_path + case, 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    # print(np.unique(label))
    
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (224 / x, 224 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            if FLAGS.model_1 == "unet_urds":
                out_main, _, _, _ = net(input)
            else:
                out_main = net(input)
            out = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 224, y / 224), order=0)
            prediction[ind] = pred
            
    if np.sum(prediction == 1)==0:
        first_metric = 0,0,0,0
    else:
        first_metric = calculate_metric_percase(prediction == 1, label == 1)
    if np.sum(prediction == 2)==0 or np.sum(label == 2)==0:
        second_metric = 0,0,0,0
    else:
        second_metric = calculate_metric_percase((prediction == 2).astype(int), (label == 2).astype(int))
    if np.sum(prediction == 3)==0:
        third_metric = 0,0,0,0
    else:
        third_metric = calculate_metric_percase(prediction == 3, label == 3)
    if np.sum(prediction == 4)==0:
        four_metric = 0,0,0,0
    else:
        four_metric = calculate_metric_percase(prediction == 4, label == 4)
    if np.sum(prediction == 5)==0:
        fifth_metric = 0,0,0,0
    else:
        fifth_metric = calculate_metric_percase(prediction == 5, label == 5)
    if np.sum(prediction == 6)==0:
        sixth_metric = 0,0,0,0
    else:
        sixth_metric = calculate_metric_percase(prediction == 6, label == 6)
    if np.sum(prediction == 7)==0:
        seventh_metric = 0,0,0,0
    else:
        seventh_metric = calculate_metric_percase(prediction == 7, label == 7)
    if np.sum(prediction == 8)==0:
        eigth_metric = 0,0,0,0
    else:
        eigth_metric = calculate_metric_percase(prediction == 8, label == 8)
        
    # # print(first_metric, second_metric, third_metric, four_metric, fifth_metric, sixth_metric, seventh_metric, eigth_metric)
    return first_metric, second_metric, third_metric, four_metric, fifth_metric, sixth_metric, seventh_metric, eigth_metric

def Inference_model1(FLAGS):
    print("——Starting the Model1 Prediction——")
    with open(FLAGS.root_path + '/test.txt', 'r') as f:image_list = f.readlines()
    image_list = sorted([item.replace('\n', '') for item in image_list])
    snapshot_path = "your_weigth_path"

    # test_save_path = "./model/test/ABD-main/ACDC_{}_{}_m_{}/{}_predictions_model/".format(FLAGS.exp, FLAGS.labeled_num, FLAGS.model_1, FLAGS.m)
    # if not os.path.exists(test_save_path):
    #     os.makedirs(test_save_path)
    net = BCP_net(in_chns=1,class_num=FLAGS.num_classes).cuda()
    # save_mode_path = os.path.join(snapshot_path, 'model1_{}_best_model.pth'.format(FLAGS.model_1))
    save_mode_path = os.path.join(snapshot_path, 'model1_unet_best_model.pth')
    
    net.load_state_dict(torch.load(save_mode_path, weights_only=False)['net'])
    print("init weight from {}".format(save_mode_path))
    net.eval()

    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    four_total=0.0
    fifth_total=0.0
    sixth_total=0.0
    seventh_total=0.0
    eigth_total=0.0
    for case in tqdm(image_list):
        first_metric, second_metric, third_metric, four_metric, fifth_metric, sixth_metric, seventh_metric, eigth_metric = test_single_volume(case, net, FLAGS)
        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)
        four_total += np.asarray(four_metric)
        fifth_total += np.asarray(fifth_metric)
        sixth_total += np.asarray(sixth_metric)
        seventh_total += np.asarray(seventh_metric)
        eigth_total += np.asarray(eigth_metric)
    avg_metric = [first_total / len(image_list), second_total / len(image_list), third_total / len(image_list), four_total / len(image_list), fifth_total / len(image_list), sixth_total / len(image_list), seventh_total / len(image_list), eigth_total / len(image_list)]
    # avg_metric = [first_total / len(image_list), second_total / len(image_list), third_total / len(image_list)]
    average = (avg_metric[0]+avg_metric[1]+avg_metric[2]+ avg_metric[3] + avg_metric[4]+avg_metric[5]+avg_metric[6]+avg_metric[7])/8
    print(avg_metric) 
    print(average)
    # with open(os.path.join(test_save_path, 'performance.txt'), 'w') as file:
    #     file.write(str(avg_metric) + '\n')
    #     file.write(str(average) + '\n')
    return avg_metric



if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    FLAGS = parser.parse_args()
    metric_model1 = Inference_model1(FLAGS)
