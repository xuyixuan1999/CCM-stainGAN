import argparse
import sys
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torch

from models import UnetClsGenerator, ResGenerator
from datasets import TestDataset

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='./datasets/MPM2HE', help='root directory of the dataset')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
parser.add_argument('--gpu_ids', type=str, default='0', help='choose gpus')
parser.add_argument('--num_worker', type=int, default=4, help='number worker of dataloader, windows must select 0')
parser.add_argument('--pretrained_model_path', type=str, default='', help='load model or not')
parser.add_argument('--outf', type=str, default='./output/test', help='root directory of the test')
opt = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids

model = UnetClsGenerator(opt.input_nc, opt.output_nc, num_residuals=1).cuda()
# model = ResGenerator(opt.input_nc, opt.output_nc).cuda()
checkpoint = torch.load(opt.pretrained_model_path, map_location='cuda')
model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model'].items()},
                      strict=True)
model.eval()

# Dataset loader
transforms_ = [ transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
testDataset = TestDataset(opt.data_root, transforms_=transforms_)
dataloader = DataLoader(testDataset, batch_size=1, 
                        shuffle=False, num_workers=opt.num_worker)


# Create output dirs if they don't exist
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

for i, batch in enumerate(dataloader):
    img = batch['img']
    name = batch['name']
    with torch.no_grad():
        output, _, _ = model(img.cuda())

    # Save image files
    output = (output.data + 1.0) * 0.5
    save_image(output, os.path.join(opt.outf, name[0]))
    sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))

sys.stdout.write('\n')