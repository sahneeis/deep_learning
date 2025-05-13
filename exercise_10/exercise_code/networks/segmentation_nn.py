"""SegmentationNN"""
import torch
import torch.nn as nn

class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = nn.ReLU()
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x



class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23, hp=None):
        super().__init__()
        self.hp = hp
        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################
        # 从 hparams 获取参数
        encoder_channels = self.hp.get("encoder_channels", [64, 128, 256])
        bottleneck_channels = self.hp.get("bottleneck_channels", 512)
        decoder_channels = self.hp.get("decoder_channels", [256, 128, 64])

        # 编码器
        self.encoder = nn.Sequential(
            ConvLayer(3, encoder_channels[0]),
            ConvLayer(encoder_channels[0], encoder_channels[1]),
            nn.MaxPool2d(2, 2),
            ConvLayer(encoder_channels[1], encoder_channels[2]),
            nn.MaxPool2d(2, 2)
        )

        # 瓶颈
        self.bottleneck = ConvLayer(encoder_channels[2], bottleneck_channels)

        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(bottleneck_channels, decoder_channels[0], kernel_size=2, stride=2),
            ConvLayer(decoder_channels[0], decoder_channels[1]),
            nn.ConvTranspose2d(decoder_channels[1], decoder_channels[2], kernel_size=2, stride=2),
            ConvLayer(decoder_channels[2], decoder_channels[2])
        )

        # 分类层
        self.classifier = nn.Conv2d(decoder_channels[2], num_classes, kernel_size=1)


        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        ########################################################################
        #                             YOUR CODE                                #  
        ########################################################################
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        x = self.classifier(x)
      
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return x

     #@property
     #def is_cuda(self):
         #"""
         #Check if model parameters are allocated on the GPU.
         #"""
         #return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self.state_dict(), path)
        # 保存模型的 state_dict
        #torch.save(model.state_dict(), "model_weights.pth")

        
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