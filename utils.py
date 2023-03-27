import datetime
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
from visdom import Visdom


def tensor2image(tensor):
    image = 127.5*(tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    return image.astype(np.uint8)

class Logger():
    def __init__(self, n_epochs, batches_epoch, start_epoch=1, env='test', port=8097):
        self.viz = Visdom(env=f'{env}', port=port)
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.set_epoch = 1
        self.epoch = start_epoch + 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}
        self.acc_windows = {}
        self.acc = {}


    def log(self, losses=None, images=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                self.losses[loss_name] = losses[loss_name].item()
            else:
                self.losses[loss_name] += losses[loss_name].item()

            if (i+1) == len(losses.keys()):
                sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name]/self.batch))
            else:
                sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name]/self.batch))

        batches_done = self.batches_epoch*(self.set_epoch - 1) + self.batch
        batches_left = self.batches_epoch*(self.n_epochs - self.epoch) + self.batches_epoch - self.batch 
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left*self.mean_period/batches_done)))
        sys.stdout.flush()

        # Draw images
        for image_name, tensor in images.items():
            if image_name not in self.image_windows:
                self.image_windows[image_name] = self.viz.image(tensor2image(tensor.data), opts={'title':image_name})
            else:
                self.viz.image(tensor2image(tensor.data), win=self.image_windows[image_name], opts={'title':image_name})

        # End of epoch
        if (self.batch % self.batches_epoch) == 0:
            # Plot losses
            for loss_name, loss in self.losses.items():
                if loss_name not in self.loss_windows:
                    self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), 
                                                                    opts={'xlabel': 'epochs', 'ylabel': loss_name, 'title': loss_name})
                else:
                    self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), win=self.loss_windows[loss_name], update='append')
                # Reset losses for next epoch
                self.losses[loss_name] = 0.0

            self.epoch += 1
            self.set_epoch += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)

def show(img):
    img = (img + 1) * 127.5
    img = Image.fromarray(img)
    img.show()
    
class CentnetLoss(nn.Module):
    def __init__(self, threshold_A=35, threshold_B=180, data_size=None):
        super(CentnetLoss, self).__init__()
        self.threshold_A = threshold_A
        self.threshold_B = threshold_B
        self.data_size = data_size
        self.total_iters = 0
        self.L1_function = nn.L1Loss()
        
    def forward(self, real_A, real_B, fake_A, fake_B):
        
        real_A_mean = torch.mean(real_A, dim=1, keepdim=True)
        real_B_mean = torch.mean(real_B, dim=1, keepdim=True)
        fake_A_mean = torch.mean(fake_A, dim=1, keepdim=True)
        fake_B_mean = torch.mean(fake_B, dim=1, keepdim=True)
        
        real_A_normal = (real_A_mean - (self.threshold_A/127.5-1))*100
        real_B_normal = (real_B_mean - (self.threshold_B/127.5-1))*100
        fake_A_normal = (fake_A_mean - (self.threshold_A/127.5-1))*100
        fake_B_normal = (fake_B_mean - (self.threshold_B/127.5-1))*100
        
        real_A_sigmoid = torch.sigmoid(real_A_normal)#.detach().numpy().astype(np.uint8)
        real_B_sigmoid = (1 - torch.sigmoid(real_B_normal))#.detach().numpy().astype(np.uint8)
        fake_A_sigmoid = torch.sigmoid(fake_A_normal)
        fake_B_sigmoid = 1 - torch.sigmoid(fake_B_normal)
        
        content_loss_A = self.L1_function( real_A_sigmoid , fake_B_sigmoid )
        content_loss_B = self.L1_function( fake_A_sigmoid , real_B_sigmoid )

        content_loss_rate = 50*np.exp(-(self.total_iters/self.data_size))
        content_loss = (content_loss_A + content_loss_B) * content_loss_rate
        
        self.total_iters += 1
        return content_loss

def save_checkpoint(model, model_name, model_root, optimizer=None, scheduler=None, epoch=None):
    if optimizer:
        if model_name=='netG_A2B':
            state = {
                    'model': model.state_dict(),
                    'epoch': epoch,
                    'optimizer': optimizer.state_dict(),
                    'schedule': scheduler.state_dict(),
                    }
        else:
            state = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'schedule': scheduler.state_dict(),
                    }
    else:
        state = {
                'model': model.state_dict(),
                }
    torch.save(state, os.path.join(model_root, '%s.pth'%model_name))

def load_checkpoint(model, model_name, model_root, optimizer=None, scheduler=None, ):
    checkpoint = torch.load(os.path.join(model_root, '%s.pth'%model_name), map_location='cuda')
    if 'optimizer' in checkpoint.keys():
        if 'epoch' in checkpoint.keys():
            model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model'].items()},
                                    strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            epoch = checkpoint['epoch']
            scheduler.load_state_dict(checkpoint['schedule'])
            return model, optimizer, scheduler, epoch
        else:
            model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model'].items()},
                                    strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['schedule'])
            return model, optimizer, scheduler
    else:
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model'].items()},
                                    strict=True)
        return model

def print_options(opt):
    """Print and save options

    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)
    # save to the disk
    file_name = os.path.join(opt.outf, 'opt.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')
