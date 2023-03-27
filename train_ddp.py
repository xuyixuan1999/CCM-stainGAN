import argparse
import datetime
import itertools
import os

import torch
import torch.distributed as dist
import torchvision.transforms as trans
from torch.autograd import Variable
from torch.utils.data import DataLoader

from datasets import ClsDataset, ImageDataset
from models import Discriminator, UnetClsGenerator
from utils import (CentnetLoss, LambdaLR, Logger, ReplayBuffer,
                   load_checkpoint, print_options, save_checkpoint,
                   weights_init_normal)

parser = argparse.ArgumentParser()
parser.add_argument('--start_epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--end_epoch', type=int, default=200, help='end epochs of training')
parser.add_argument('--batch_size', type=int, default=4, help='size of the batches')
parser.add_argument('--batch_size_cls', type=int, default=16, help='size of the batches of class')
parser.add_argument('--data_root', type=str, default='../datasets/MPM2HE-256', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--num_classes', type=int, default=9,help='Number of categories')
parser.add_argument('--decay_epoch', type=int, default=100,help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--gpu_ids', type=str, default='0', help='choose gpus')
parser.add_argument('--num_worker', type=int, default=4, help='number worker of dataloader')
parser.add_argument('--threshold_A', type=int, default=35, help='threshold of A')
parser.add_argument('--threshold_B', type=int, default=180, help='threshold of B')
parser.add_argument('--outf', type=str, default='./output/', help='root directory of the models')
parser.add_argument('--pretrained_model_path', type=str, default='', help='load model or not')
parser.add_argument('--env', type=str, default='ccm', help='environment name of visdom')
parser.add_argument('--local_rank', default=-1, type=int, help='local device id on current node')


opt = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
n_gpus = len(opt.gpu_ids.split(','))

dist.init_process_group(backend='nccl', world_size=n_gpus, rank=opt.local_rank)
torch.cuda.set_device(opt.local_rank)

transforms_ = [
    trans.Resize(int(opt.size * 1.12), trans.InterpolationMode.BICUBIC),
    trans.CenterCrop(opt.size),
    trans.RandomHorizontalFlip(),
    trans.RandomVerticalFlip(),
    trans.ToTensor(),
    trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]
transforms_cls = [
    trans.Resize(int(opt.size * 1.12), trans.InterpolationMode.BICUBIC),
    trans.RandomCrop(opt.size),
    trans.RandomHorizontalFlip(),
    trans.RandomVerticalFlip(),
    trans.ToTensor(),
    trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]

dataset = ImageDataset(opt.data_root, transforms_=transforms_, batch_size=opt.batch_size)
train_sample = torch.utils.data.distributed.DistributedSampler(dataset)

dataset_cls = ClsDataset(opt.data_root, transforms_=transforms_cls, batch_size=opt.batch_size_cls)
cls_sample = torch.utils.data.distributed.DistributedSampler(dataset_cls)

dataloader = DataLoader(dataset, batch_size=opt.batch_size, #shuffle=True, 
                        num_workers=opt.num_worker, drop_last=True, sampler=train_sample)
dataloader_cls = DataLoader(dataset_cls, batch_size=opt.batch_size_cls, #shuffle=True, 
                            num_workers=opt.num_worker, drop_last=True, sampler=cls_sample)

time_str = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
opt.outf = opt.outf + time_str

if not os.path.exists(opt.outf) and opt.local_rank == 0:
    os.makedirs(opt.outf)
    print_options(opt)


###### define networks ######
#   Generator
netG_A2B = UnetClsGenerator(opt.input_nc, opt.output_nc, opt.num_classes, num_residuals=1).cuda()
netG_B2A = UnetClsGenerator(opt.output_nc, opt.input_nc, opt.num_classes, num_residuals=1).cuda()

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
# criterion_identity = torch.nn.L1Loss()
criterion_class = torch.nn.CrossEntropyLoss()

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
    netG_A2B.cuda(opt.local_rank)
    netG_B2A.cuda(opt.local_rank)
    netD_A.cuda(opt.local_rank)
    netD_B.cuda(opt.local_rank)
    
if torch.cuda.device_count() > 1:
    netG_A2B = torch.nn.parallel.DistributedDataParallel(netG_A2B.cuda(opt.local_rank), device_ids=[opt.local_rank], find_unused_parameters=False)
    netG_B2A = torch.nn.parallel.DistributedDataParallel(netG_B2A.cuda(opt.local_rank), device_ids=[opt.local_rank], find_unused_parameters=False)
    netD_A = torch.nn.parallel.DistributedDataParallel(netD_A.cuda(opt.local_rank), device_ids=[opt.local_rank], find_unused_parameters=False)
    netD_B = torch.nn.parallel.DistributedDataParallel(netD_B.cuda(opt.local_rank), device_ids=[opt.local_rank], find_unused_parameters=False)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()


###### Loss plot ######
if opt.local_rank == 0:
    logger = Logger(opt.end_epoch, len(dataloader), start_epoch, '%s' % (opt.env))
#######################

target_real = torch.ones((opt.batch_size, 1), dtype=torch.float32, requires_grad=False).cuda(opt.local_rank)
target_fake = torch.zeros((opt.batch_size, 1), dtype=torch.float32, requires_grad=False).cuda(opt.local_rank)
total_iters = len(dataloader) * start_epoch
###### Training ######
for epoch in range(start_epoch, opt.end_epoch):
    train_sample.set_epoch(epoch)
    cls_sample.set_epoch(epoch)
    for i, batch_data in enumerate(zip(itertools.cycle(dataloader_cls), dataloader)):
        cls_batch, batch = batch_data
        # class input
        real_cls_A = Variable(cls_batch['img_A']).cuda(opt.local_rank)
        real_cls_B = Variable(cls_batch['img_B']).cuda(opt.local_rank)
        real_label_A = Variable(cls_batch['label_A']).long().cuda(opt.local_rank)
        real_label_B = Variable(cls_batch['label_B']).long().cuda(opt.local_rank)
        # model input
        real_A = Variable(batch['A']).cuda(opt.local_rank)
        real_B = Variable(batch['B']).cuda(opt.local_rank)
        
        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()
        
        # cls loss
        cls_out_A = netG_A2B(real_cls_A, cls=True)
        loss_cls_A = criterion_class(cls_out_A, real_label_A)
        cls_out_B = netG_B2A(real_cls_B, cls=True)
        loss_cls_B = criterion_class(cls_out_B, real_label_B)
        loss_cls = (loss_cls_A + loss_cls_B) * 0.5
        
        # GAN loss
        fake_B, features_ra, cls_ra = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

        fake_A, features_rb, cls_rb = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

        # Cycle loss
        recovered_A, features_fb, cls_fb = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * 10.0

        recovered_B, features_fa, cls_fa = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * 10.0

        # class same loss
        loss_cls_ABA = criterion_cycle(cls_ra, cls_fb)
        loss_cls_BAB = criterion_cycle(cls_rb, cls_fa)
        loss_cls_same = (loss_cls_ABA + loss_cls_BAB)

        # base loss
        loss_base_A = criterion_cycle(features_ra, features_fb)
        loss_base_B = criterion_cycle(features_rb, features_fa)
        loss_base = (loss_base_A + loss_base_B)

        # content loss 
        loss_content = criterion_content(real_A, real_B, fake_A, fake_B)
        
        loss_G = loss_GAN_A2B + loss_GAN_B2A + \
                 loss_cycle_BAB + loss_cycle_ABA + \
                 loss_cls + loss_base + loss_content + loss_cls_same
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
        if opt.local_rank == 0:
            logger.log({'loss_G': loss_G,
                        'loss_content': (loss_content),
                        'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                        'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB),
                        'loss_base': loss_base,
                        'loss_class': (loss_cls),
                        'loss_cls_same': loss_cls_same,
                        'loss_D': (loss_D_A + loss_D_B),
                        },
                    images={'real_A': real_A, 'fake_B': fake_B, 'real_B': real_B, 'fake_A': fake_A})
        
        ###################################
        # save model per half an epoch
        if (i + 1) % (len(dataloader) // 5) == 0 and opt.local_rank == 0:
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
    if not os.path.exists(modelroot) and opt.local_rank == 0:
        os.makedirs(modelroot)
    if opt.local_rank == 0:
        # save netG_A2B
        save_checkpoint(netG_A2B, 'netG_A2B', modelroot, optimizer_G, lr_scheduler_G, epoch+1,)
        # save netG_B2A
        save_checkpoint(netG_B2A, 'netG_B2A', modelroot)
        # save netD_A
        save_checkpoint(netD_A, 'netD_A', modelroot, optimizer_D_A, lr_scheduler_D_A)
        # save netD_B
        save_checkpoint(netD_B, 'netD_B', modelroot, optimizer_D_B, lr_scheduler_D_B)
