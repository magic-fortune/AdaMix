import argparse
import os
import shutil
import h5py
import numpy as np
import SimpleITK as sitk
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
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/train_PROMISE12', help='experiment_name')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--model_1', type=str,
                    default='unet', help='model1_name')
parser.add_argument('--model_2', type=str,
                    default='swin_unet', help='model2_name')
parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=3,
                    help='labeled data')
parser.add_argument('--image_size', type=list, default=[224, 224],
                    help='patch size of network input')
parser.add_argument('--cfg', type=str,
                    default="../code/configs/swin_tiny_patch4_window7_224_lite.yaml",
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
args = parser.parse_args()  
config = get_config(args)

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    asd = metric.binary.asd(pred, gt)
    return dice, asd

def test_single_volume(case, net):
    np_data_path = os.path.join("../data/promise12", 'npy_image')
    img = np.load(os.path.join(np_data_path, '{}.npy'.format(case)))
    mask = np.load(os.path.join(np_data_path, '{}_segmentation.npy'.format(case)))
    prediction = np.zeros_like(mask)
    for ind in range(img.shape[0]):
        slice = img[ind, :, :]
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            prediction[ind] = out
    if np.sum(prediction == 1) == 0:
        first_metric = 0 ,0
    else:
        first_metric = calculate_metric_percase(prediction == 1, mask == 1)
    return first_metric

def Inference_model1(FLAGS):
    print("——Starting the Model1 Prediction——")
    with open("../data/promise12" + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])
    
    snapshot_path = "./model/Cross_Teaching/PROMISE12_train_PROMISE12_3_vit"
    test_save_path = "./model/test/Cross_Teaching/PROMISE12_{}_{}_m=2/{}_predictions_model/".format(FLAGS.exp, FLAGS.labeled_num, FLAGS.model_1)
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)
    net = net_factory(net_type=FLAGS.model_1, in_chns=1,class_num=FLAGS.num_classes)
    save_mode_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model_1))
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    first_total = 0.0
    for case in tqdm(image_list):
        first_metric = test_single_volume(case, net)
        first_total += np.asarray(first_metric)
    avg_metric = [first_total / len(image_list)]
    print(avg_metric)
    with open(os.path.join(test_save_path, 'performance.txt'), 'w') as file:
        file.write(str(avg_metric) + '\n')
    return avg_metric


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    FLAGS = parser.parse_args()
    metric_model1 = Inference_model1(FLAGS)
    # metric_model2 = Inference_model2(FLAGS)