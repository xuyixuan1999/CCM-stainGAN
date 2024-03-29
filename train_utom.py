import argparse
import datetime
import itertools
import os

import torch
import torchvision.transforms as trans
from torch.autograd import Variable
from torch.utils.data import DataLoader

from datasets import ImageDataset
from models import Discriminator, ResGenerator
from utils import (LambdaLR, Logger, ReplayBuffer, CentnetLoss, print_options,
                   weights_init_normal, save_checkpoint, load_checkpoint)

parser = argparse.ArgumentParser()
parser.add_argument('--start_epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--end_epoch', type=int, default=200, help='end epochs of training')
parser.add_argument('--batch_size', type=int, default=5, help='size of the batches')
parser.add_argument('--data_root', type=str, default='./datasets/MPM2HE', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100,help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--gpu_ids', type=str, default='5', help='choose gpus')
parser.add_argument('--num_worker', type=int, default=4, help='number worker of dataloader')
parser.add_argument('--threshold_A', type=int, default=35, help='threshold of A')
parser.add_argument('--threshold_B', type=int, default=180, help='threshold of B')
parser.add_argument('--outf', type=str, default='./output/', help='root directory of the models')
parser.add_argument('--pretrained_model_path', type=str, default='', help='load model or not')
parser.add_argument('--env', type=str, default='utom', help='environment name of visdom')

opt = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
n_gpus = len(opt.gpu_ids.split(','))

transforms_ = [
    trans.Resize(int(opt.size * 1.12), trans.InterpolationMode.BICUBIC),
    trans.CenterCrop(opt.size),
    trans.RandomHorizontalFlip(),
    trans.RandomVerticalFlip(),
    trans.ToTensor(),
    trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]

dataset = ImageDataset(opt.data_root, transforms_=transforms_, batch_size=opt.batch_size)
dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, 
                        num_workers=opt.num_worker, drop_last=True)

time_str = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
opt.outf = opt.outf + time_str

if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)
    print_options(opt)


###### define networks ######
#   Generator
netG_A2B = ResGenerator(opt.input_nc, opt.output_nc).cuda()
netG_B2A = ResGenerator(opt.output_nc, opt.input_nc).cuda()

#   Discriminator
netD_A = Discriminator(opt.input_nc).cuda()
netD_B = Discriminator(opt.output_nc).cuda()

#   init model
netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)
   
#   Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_content = CentnetLoss(opt.threshold_A, opt.threshold_B, len(dataloader))

#   Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                               lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G,lr_lambda=LambdaLR(opt.end_epoch, opt.start_epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_A, lr_lambda=LambdaLR(opt.end_epoch, opt.start_epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_B, lr_lambda=LambdaLR(opt.end_epoch, opt.start_epoch, opt.decay_epoch).step)

###### load pretrained model ######
start_epoch = opt.start_epoch
resume_file = opt.pretrained_model_path
if resume_file:
    netG_A2B, optimizer_G, lr_scheduler_G, start_epoch = load_checkpoint(
        netG_A2B, 'netG_A2B', resume_file, optimizer_G, lr_scheduler_G)
        
    netG_B2A = load_checkpoint(netG_B2A, 'netG_B2A', resume_file)
    
    netD_A, optimizer_D_A, lr_scheduler_D_A = load_checkpoint(
        netD_A, 'netD_A', resume_file, optimizer_D_A, lr_scheduler_D_A)
        
    netD_B, optimizer_D_B, lr_scheduler_D_B = load_checkpoint(
        netD_B, 'netD_B', resume_file, optimizer_D_B, lr_scheduler_D_B)


if torch.cuda.is_available():
    netG_A2B.cuda()
    netG_B2A.cuda()
    netD_A.cuda()
    netD_B.cuda()
    
if torch.cuda.device_count() > 1:
    netG_A2B = torch.nn.DataParallel(netG_A2B)
    netG_B2A = torch.nn.DataParallel(netG_B2A)
    netD_A = torch.nn.DataParallel(netD_A)
    netD_B = torch.nn.DataParallel(netD_B)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()


###### Loss plot ######
logger = Logger(opt.end_epoch, len(dataloader),
                start_epoch, '%s' % (opt.env), opt.port)
#######################

target_real = torch.ones((opt.batch_size, 1), dtype=torch.float32, requires_grad=False).cuda()
target_fake = torch.zeros((opt.batch_size, 1), dtype=torch.float32, requires_grad=False).cuda()
total_iters = len(dataloader) * start_epoch
###### Training ######
for epoch in range(start_epoch, opt.end_epoch):
    for i, batch in enumerate(dataloader):
        # model input
        real_A = Variable(batch['A']).cuda()
        real_B = Variable(batch['B']).cuda()
        
        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()
        
        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * 10.0

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * 10.0

        # content loss 
        loss_content = criterion_content(real_A, real_B, fake_A, fake_B)
        
        loss_G = loss_GAN_A2B + loss_GAN_B2A + \
                 loss_cycle_BAB + loss_cycle_ABA + loss_content 
                 
        loss_G.backward()
        
        optimizer_G.step()
        ###################################
        
        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake) * 0.5
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################
        
        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake) * 0.5
        loss_D_B.backward()

        optimizer_D_B.step()
        ###################################

        logger.log({'loss_G': loss_G,
                    'loss_content': (loss_content),
                    'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                    'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB),
                    'loss_D': (loss_D_A + loss_D_B),
                    },
                   images={'real_A': real_A, 'fake_B': fake_B, 'real_B': real_B, 'fake_A': fake_A})
        
        ###################################
        # save model per half an epoch
        if (i + 1) % (len(dataloader) // 5) == 0:
            model_root = os.path.join(opt.outf, 'temp')
            if not os.path.exists(model_root):
                os.makedirs(model_root)
            # save netG_A2B
            save_checkpoint(netG_A2B, 'netG_A2B', model_root, optimizer_G, lr_scheduler_G, epoch)
            # save netG_A2B
            save_checkpoint(netG_B2A, 'netG_B2A', model_root)
            # save netD_A
            save_checkpoint(netD_A, 'netD_A', model_root, optimizer_D_A, lr_scheduler_D_A)
            # save netD_B
            save_checkpoint(netD_B, 'netD_B', model_root, optimizer_D_B, lr_scheduler_D_B)
        ###################################
        
    # update the learning rate       
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()   
     
    ################################### 
    modelroot = os.path.join(opt.outf, 'epoch' + str(epoch+1))        
    if not os.path.exists(modelroot):
        os.makedirs(modelroot)
    # save netG_A2B
    save_checkpoint(netG_A2B, 'netG_A2B', modelroot, optimizer_G, lr_scheduler_G, epoch+1)
    # save netG_B2A
    save_checkpoint(netG_B2A, 'netG_B2A', modelroot)
    # save netD_A
    save_checkpoint(netD_A, 'netD_A', modelroot, optimizer_D_A, lr_scheduler_D_A)
    # save netD_B
    save_checkpoint(netD_B, 'netD_B', modelroot, optimizer_D_B, lr_scheduler_D_B)