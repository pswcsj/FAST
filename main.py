import torch
import argparse
import os
import sys
from mmcv import Config
import mmcv
from dataset import build_data_loader
from models import build_model
from models.utils import fuse_module, rep_model_convert
from utils import ResultFormat, AverageMeter
from dataset.utils import get_img
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from dataset.utils import scale_aligned_short, center_crop

from mmcv.cnn import get_model_complexity_info
import logging
import warnings
import torch.nn as nn
warnings.filterwarnings('ignore')
import json

short_size = 640
pooling_size=9
read_type='pil'
# class FAST_IC17MLT(data.Dataset):
#     def __init__(self, img_size=None, short_size=640, pooling_size=9, with_rec=False, read_type='pil'):
#         self.img_size = img_size if (img_size is None or isinstance(img_size, tuple)) else (img_size, img_size)
#         self.pooling_size = pooling_size
#         self.short_size = short_size
#         self.with_rec = with_rec
#         self.read_type = read_type
#
#         self.pad = nn.ZeroPad2d(padding=(pooling_size - 1) // 2)
#         self.pooling = nn.MaxPool2d(kernel_size=pooling_size, stride=1)
#         self.overlap_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
#
#         self.img_paths = []
#         self.gt_paths = []
#
#
#         self.voc, self.char2id, self.id2char = get_vocabulary('LOWERCASE')
#         self.max_word_num = 200
#         self.max_word_len = 32
#
#     def __len__(self):
#         return len(self.img_paths)
#
#     def prepare_test_data(self, index):
#         img = get_img("aa.jpg", self.read_type)
#         img_meta = dict(
#             org_img_size=np.array(img.shape[:2])
#         )
#
#         img = scale_aligned_short(img, self.short_size)
#         img_meta.update(dict(
#             img_size=np.array(img.shape[:2])
#         ))
#
#         img = Image.fromarray(img)
#         img = img.convert('RGB')
#         img = transforms.ToTensor()(img)
#         img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
#
#         data = dict(
#             imgs=img,
#             img_metas=img_meta
#         )
#
#         return data
#
#     def __getitem__(self, index):
#         return self.prepare_test_data(index)


def test(test_loader, model, cfg):
    rf = ResultFormat(cfg.data.test.type, cfg.test_cfg.result_path)

    if cfg.report_speed:
        speed_meters = dict(
            backbone_time=AverageMeter(1000 // args.batch_size),
            neck_time=AverageMeter(1000 // args.batch_size),
            det_head_time=AverageMeter(1000 // args.batch_size),
            post_time=AverageMeter(1000 // args.batch_size),
            total_time=AverageMeter(1000 // args.batch_size)
        )
    results = dict()

    for idx, data in enumerate(test_loader):
        print('Testing %d/%d\r' % (idx, len(test_loader)), flush=True, end='')
        logging.info('Testing %d/%d\r' % (idx, len(test_loader)))
        # prepare input
        if not args.cpu:
            data['imgs'] = data['imgs'].cuda(non_blocking=True)
        data.update(dict(cfg=cfg))
        # forward
        with torch.no_grad():
            outputs = model(**data)

        if cfg.report_speed:
            report_speed(model, data, speed_meters, cfg.batch_size)
            continue

        # save result
        image_names = data['img_metas']['filename']
        for index, image_name in enumerate(image_names):
            rf.write_result(image_name, outputs['results'][index])
            results[image_name] = outputs['results'][index]

    if not cfg.report_speed:
        results = json.dumps(results)
        with open('outputs/output.json', 'w', encoding='utf-8') as json_file:
            json.dump(results, json_file, ensure_ascii=False)
            print("write json file success!")


def model_structure(model):
    blank = ' '
    print('-' * 90)
    print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
          + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 90)
    num_para = 0
    type_size = 1  ##如果是浮点数就是4

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30:
            key = key + (30 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('-' * 90)


def main(args):
    # cfg = Config.fromfile(args.config)
    cfg = Config.fromfile("config/fast/tt/fast_base_tt_800_finetune_ic17mlt.py")
    args.checkpoint = "pretrained/fast_base_tt_800_finetune_ic17mlt.pth"
    for d in [cfg, cfg.data.test]:
        d.update(dict(
            report_speed=args.report_speed,
        ))
    if args.min_score is not None:
        cfg.test_cfg.min_score = args.min_score
    if args.min_area is not None:
        cfg.test_cfg.min_area = args.min_area

    cfg.batch_size = args.batch_size

    # data loader
    model = build_model(cfg.model)

    if not args.cpu:
        model = model.cuda()

    if args.checkpoint is not None:
        if os.path.isfile(args.checkpoint):
            print("Loading model and optimizer from checkpoint '{}'".format(args.checkpoint))
            logging.info("Loading model and optimizer from checkpoint '{}'".format(args.checkpoint))
            sys.stdout.flush()
            checkpoint = torch.load(args.checkpoint)

            if not args.ema:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint['ema']

            d = dict()
            for key, value in state_dict.items():
                tmp = key.replace("module.", "")
                d[tmp] = value
            model.load_state_dict(d)
        else:
            print("No checkpoint found at '{}'".format(args.checkpoint))
            raise

    model = rep_model_convert(model)

    # fuse conv and bn
    model = fuse_module(model)

    # 모델 구조 출력
    if args.print_model:
        model_structure(model)

    # flops, params = get_model_complexity_info(model, (3, 1280, 864))
    # flops, params = get_model_complexity_info(model, (3, 1200, 800))
    # flops, params = get_model_complexity_info(model, (3, 1344, 896))
    # flops, params = get_model_complexity_info(model, (3, 960, 640))
    # flops, params = get_model_complexity_info(model, (3, 768, 512))
    # flops, params = get_model_complexity_info(model, (3, 672, 448))
    # flops, params = get_model_complexity_info(model, (3, 480, 320))
    # print(flops, params)

    model.eval()

    img = get_img("aa.jpg", read_type)
    img = scale_aligned_short(img, short_size)
    img = Image.fromarray(img)
    img = img.convert('RGB')
    img = transforms.ToTensor()(img)
    img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

    with torch.no_grad():
        print(model(img))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', nargs='?', type=str, default=None)
    parser.add_argument('--report-speed', action='store_true')
    parser.add_argument('--print-model', action='store_true')
    parser.add_argument('--min-score', default=None, type=float)
    parser.add_argument('--min-area', default=None, type=int)
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--worker', default=4, type=int)
    parser.add_argument('--ema', action='store_true')
    parser.add_argument('--cpu', action='store_true')

    args = parser.parse_args()
    mmcv.mkdir_or_exist("./speed_test")
    config_name = os.path.basename(args.config)
    logging.basicConfig(filename=f'./speed_test/{config_name}.txt', level=logging.INFO)

    main(args)
