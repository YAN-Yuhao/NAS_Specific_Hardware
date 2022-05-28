import os
import sys
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import psutil
import time
from torch.autograd import Variable
from model_search import Network


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='EXP/weights.pt', help='path of pretrained model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

CIFAR_CLASSES = 10

def count_time(func):
    def int_time(a,arg,b):
        start_time = time.time()
        func(a,arg,b)
        over_time = time.time()
        total_time = (over_time - start_time)*1000
        print("time cost %s" % total_time)
    return int_time

def count_info(func):
    def float_info(a,arg,b):
        pid = os.getpid()
        p = psutil.Process(pid)
        info_start = p.memory_full_info().uss/1024
        func(a,arg,b)
        info_end=p.memory_full_info().uss/1024
        print("memory cost "+str(info_end-info_start)+"KB")
    return float_info

def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()

  genotype = eval("genotypes.%s" % args.arch)
  model = Network(16, CIFAR_CLASSES, 8, criterion)
  model = model.cuda()
  utils.load(model, args.model_path)


  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  _, test_transform = utils._data_transforms_cifar10(args)
  test_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=test_transform)

  test_queue = torch.utils.data.DataLoader(
      test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

  model.drop_path_prob = args.drop_path_prob
  infer(test_queue, model, criterion)

@count_time
@count_info
def infer(test_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(test_queue):
    input = Variable(input, volatile=True).cuda()
    target = Variable(target, volatile=True).cuda(non_blocking=True)

    logits = model(input)
    loss = criterion(logits, target)



if __name__ == '__main__':
  main() 

