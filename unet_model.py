""" Full assembly of the parts to form the complete network """

from .unet_parts import *

class UNet_EP(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_EP, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        
        )

        factor = 2 if bilinear else 1

        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(512, 1024 // factor, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024 // factor),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024 // factor, 1024 // factor, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024 // factor),
            nn.ReLU(inplace=True),
        )
        
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

        self.res = []

        self.extractor = [
            *self.inc, *self.down1, *self.down2, *self.down3, *self.down4
        ]

        for i in range(len(self.extractor)):
            self.extractor[i].register_forward_hook(self.get_activation())

    def get_activation(self):
        def hook(model, input, output):
            self.res.append(output.squeeze(0).cpu().numpy())
        return hook

    def getLayers(self):
        # all model parts in order in one list 
        return self.extractor
    
    def getFeatures(self):
        # all feature outputs in order in one list 
        return self.res

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    

# if __name__ == '__main__':
#     net = UNet_EP(3, 2)
#     data = torch.rand(1, 3, 512, 512)
#     with torch.no_grad():
#         _ = net(data)