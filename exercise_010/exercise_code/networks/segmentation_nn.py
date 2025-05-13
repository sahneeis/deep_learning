"""SegmentationNN"""
import torch
import torch.nn as nn


class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


# class ModifiedMobileNetV2(nn.Module):
#     def __init__(self, input_channels=3, num_classes=1000, freeze_layers=True):
#         super(ModifiedMobileNetV2, self).__init__()
        
#         self.mobilenet_v2 = mobilenet_v2(pretrained=True)
        
#         self.mobilenet_v2.features[0][0] = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        
#         self.mobilenet_v2.classifier[1] = nn.Linear(1280, num_classes)
        
#         if freeze_layers:
#             self.freeze_layers()

#     def freeze_layers(self):
#         for param in self.mobilenet_v2.features.parameters():
#             param.requires_grad = False

#         for param in self.mobilenet_v2.classifier[1].parameters():
#             param.requires_grad = True

#     def forward(self, x):
#         return self.mobilenet_v2(x)  


class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23, hp=None, ch=None, mobilenet=None): 
        super().__init__()
        self.hparams = hp
        self.ch = ch

        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        
        self.mobile_net = mobilenet.features
        self.cnn = nn.Sequential(
            ConvLayer(ch[0], ch[1]),
            ConvLayer(ch[1], ch[2]),
            nn.MaxPool2d(2, 2),
            
            ConvLayer(ch[2], ch[3]),
            ConvLayer(ch[3], ch[4]),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(ch[4], ch[5], kernel_size=1, padding=0)
        )
        
        self.upsampling = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )
        
        self.adjust = nn.Sequential(
            nn.Conv2d(ch[5], num_classes, kernel_size=1, padding=0), # [32, 23, 8, 8]
            nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Upsample(scale_factor=2, mode='bilinear'), # 256*256
            nn.Conv2d(num_classes, num_classes, kernel_size=17, padding=0) 
            # (256-k)/s + 1 = 240, assume s = 1, k = 17
        )

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        
        x = self.mobile_net(x)
        x = self.cnn(x)
        x = self.upsampling(x)
        x = self.adjust(x)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    # @property
    # def is_cuda(self):
    #     """
    #     Check if model parameters are allocated on the GPU.
    #     """
    #     return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

        
class DummySegmentationModel(nn.Module):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()

if __name__ == "__main__":
    from torchinfo import summary
    summary(SegmentationNN(), (1, 3, 240, 240), device="cpu")