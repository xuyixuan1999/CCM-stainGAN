import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)
    
class ResGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(ResGenerator, self).__init__()

        # Initial convolution block       
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(3):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(3):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
    
class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]
        model += [nn.Conv2d(256, 512, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(512),
                  nn.LeakyReLU(0.2, inplace=True)]

        # # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1) # return x
    
class UnetGenerator(nn.Module):
    '''
        U-net 
    '''
    def __init__(self, input_nc, output_nc, num_downs=8, ngf=64):
        super(UnetGenerator, self).__init__()
        unet_block = UnetBlock(8*ngf, 8*ngf, input_nc=None, innermost=True)
        for i in range(num_downs-6):
            unet_block = UnetBlock(8*ngf, 8*ngf, submodule=unet_block)
        unet_block = UnetBlock(ngf*4, ngf*8, submodule=unet_block)
        unet_block = UnetBlock(ngf*2, ngf*4, submodule=unet_block)
        unet_block = UnetBlock(ngf, ngf*2, submodule=unet_block)
        self.model = UnetBlock(output_nc, ngf, input_nc, submodule=unet_block, outermost=True)


    def forward(self, x):
        return self.model(x)

class UnetBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, innermost=False, outermost=False, use_bias=True):
        super(UnetBlock, self).__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.InstanceNorm2d(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = nn.InstanceNorm2d(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc*2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)   ##   inner_nc*2 是因为cat操作后通道数翻倍
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc*2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)  ##   inner_nc*2 是因为cat操作后通道数翻倍
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)
        
class UnetDown(nn.Module):
    def __init__(self, in_features, out_features, use_bias=False, num_residual=1):
        super(UnetDown, self).__init__()
        down = nn.Sequential(
            nn.Conv2d(in_features, out_features, 3, stride=2, padding=1, bias=use_bias),
            nn.InstanceNorm2d(out_features),
            nn.LeakyReLU(1e-2, inplace=True)
            )
        blk = []
        for _ in range(num_residual):
            blk += [ResidualBlock(out_features)]
        self.model = nn.Sequential(down, *blk)
        
    def forward(self, x):
        return self.model(x)

class UnetUp(nn.Module):
    def __init__(self, in_features, out_features, use_bias=False, num_residual=1):
        super(UnetUp, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, 
                               output_padding=1, bias=use_bias),
            nn.InstanceNorm2d(out_features),
            nn.LeakyReLU(1e-2, inplace=True)
        )
        self.merge = nn.Sequential(
            nn.Conv2d(out_features*2, out_features, 3, stride=1, padding=1, bias=use_bias),
            nn.InstanceNorm2d(out_features),
            nn.LeakyReLU(1e-2, inplace=True)
        )
        blk = []
        for _ in range(num_residual):
            blk += [ResidualBlock(out_features)]
        self.blk = nn.Sequential(*blk)
        
    def forward(self, x, skip_x):
        x = self.up(x)
        
        dif_x = x.size()[2] - skip_x.size()[2]
        dif_y = x.size()[3] - skip_x.size()[3]
        skip_x = F.pad(skip_x, [dif_x//2, dif_x//2,
                                dif_y//2, dif_y//2])
        x = self.merge(torch.cat([x, skip_x], dim=1))
        return self.blk(x)
    
class ClsModel(nn.Module):
    def __init__(self, ngf, eps, num_classes):
        super(ClsModel, self).__init__()
        cls_model = [

            nn.Conv2d(ngf*4, ngf*8, 3, padding=1, stride=2),
            nn.InstanceNorm2d(ngf*8),
            nn.LeakyReLU(eps, inplace=True),
            ResidualBlock(ngf*8),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf*8, ngf*8, 3, padding=1, stride=2),
            nn.InstanceNorm2d(ngf*8),
            nn.LeakyReLU(eps, inplace=True),
            ResidualBlock(ngf*8),
            
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(ngf*8, num_classes),
            ]
        
        self.cls = nn.Sequential(*cls_model)
        
    def forward(self, x):
        x = self.cls(x)
        return x
        
class UnetClsGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_classes=9, num_residuals=1, num_exact=8, ngf=64, eps=1e-2):
        super(UnetClsGenerator, self).__init__()
        self.inconv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf//2, 7),
            nn.InstanceNorm2d(ngf//2),
            nn.LeakyReLU(eps, inplace=True),
            nn.Conv2d(ngf//2, ngf//2, 3, padding=1),
            nn.InstanceNorm2d(ngf//2),
            nn.LeakyReLU(eps, inplace=True)
        )
        
        # 1.down sample [b, 64, H/2, W/2]
        self.down1 = UnetDown(ngf//2, ngf, True, num_residuals)
        # 2.down sample [b, 128, H/4, W/4]
        self.down2 = UnetDown(ngf, ngf*2, True, num_residuals)
        # 3.down sample [b, 256, H/8, W/8]
        self.down3 = UnetDown(ngf*2, ngf*4, True, num_residuals)
        # 4.down sample [b, 512, H/16, W/16]
        # self.down4 = UnetDown(ngf*4, ngf*8, True, 0)
        
        # cls
        self.cls = ClsModel(ngf, eps, num_classes)
        
        # feature decoder
        feature_de = []
        for _ in range(num_exact//2):
            feature_de += [ResidualBlock(ngf*4)]
        self.decoder = nn.Sequential( *feature_de)
        
        # feature encoder
        feature_en = []
        for _ in range(num_exact//2):
            feature_en += [ResidualBlock(ngf*4)]
        self.encoder = nn.Sequential(*feature_en)
        
        # 4.up sample [b, 256, H/8, W/8]
        # self.up4 = UnetUp(ngf*8, ngf*4, True, num_residuals)
        # 3.up sample [b, 128, H/4, W/4]
        self.up3 = UnetUp(ngf*4, ngf*2, True, num_residuals)
        # 2.up sample [b, 64, H/2, W/2]
        self.up2 = UnetUp(ngf*2, ngf, True, num_residuals)
        # 1.up sample [b, 32, H, W]
        self.up1 = UnetUp(ngf, ngf//2, True, num_residuals)
        
        # ouput1 [b, 3, H, W]
        self.outconv = nn.Sequential(
            nn.Conv2d(ngf//2, ngf//2, 3, padding=1),
            nn.InstanceNorm2d(ngf//2),
            nn.LeakyReLU(eps, inplace=True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf//2, output_nc, 7),
            nn.Tanh()
        )
        
    def forward(self, x, cls=False):
        x0 = self.inconv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        cls_out = self.cls(x3)
        
        if cls:
            return cls_out
        destain_features = self.decoder(x3)
        exact_features = self.encoder(destain_features)

        x = self.up3(exact_features, x2)
        x = self.up2(x, x1)
        x = self.up1(x, x0)
        x = self.outconv(x)
        
        return x, destain_features, cls_out
    
if __name__ == "__main__":
    input = torch.randn(8, 3, 256, 256).requires_grad_(True)
    # model = UnetClsGenerator(3, 3, 9, 1, 8, 64, 1e-2)
    model = Discriminator(3)
    # output, de_fea_ra, cls_ra = model(input)
    output = model(input)
    # print(output.size(), de_fea_ra.size(), cls_ra.size())
    print(output.shape)
