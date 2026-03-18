# Define EfficientNetB0 model architecture
from efficientnet_pytorch import EfficientNet
import timm
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from sklearn.metrics import accuracy_score
from PIL import Image
import io


class EfficientNetB0(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetB0, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)

    def forward(self, x):
        return self.efficientnet(x)


# Define XceptionNet model architecture
class XceptionNet(nn.Module):
    def __init__(self, num_classes):
        super(XceptionNet, self).__init__()
        self.xceptionnet = timm.create_model('xception', pretrained=True)

        # Modify the output layer for binary classification
        num_ftrs = self.xceptionnet.fc.in_features
        self.xceptionnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.xceptionnet(x)
    

    # Define EfficientNetB7 model architecture
class EfficientNetB7(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetB7, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b7', num_classes=num_classes)

    def forward(self, x):
        return self.efficientnet(x)

# Define PPG model architecture
class Attention_mask(nn.Module):
    def __init__(self):
        super(Attention_mask, self).__init__()

    def forward(self, x):
        xsum = torch.sum(x, dim=2, keepdim=True)
        xsum = torch.sum(xsum, dim=3, keepdim=True)
        xshape = tuple(x.size())
        return x / xsum * xshape[2] * xshape[3] * 0.5

    def get_config(self):
        """May be generated manually. """
        config = super(Attention_mask, self).get_config()
        return config

class DeepPhys(nn.Module):

    def __init__(self, in_channels=3, nb_filters1=32, nb_filters2=64, kernel_size=3, dropout_rate1=0.25,
                 dropout_rate2=0.5, pool_size=(2, 2), nb_dense=128, img_size=36):
        """Definition of DeepPhys.
        Args:
        in_channels: the number of input channel. Default: 3
        img_size: height/width of each frame. Default: 36.
        Returns:
        DeepPhys model.
        """
        super(DeepPhys, self).__init__()

        # self.final_dense_1 = nn.Linear(16384, 128, bias=True)

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.pool_size = pool_size
        self.nb_filters1 = nb_filters1
        self.nb_filters2 = nb_filters2
        self.nb_dense = nb_dense

        # Motion branch convs
        self.motion_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1),
                                    bias=True)
        self.motion_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, bias=True)
        self.motion_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1),
                                    bias=True)
        self.motion_conv4 = nn.Conv2d(self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, bias=True)

        # Apperance branch convs
        self.apperance_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size,
                                        padding=(1, 1), bias=True)
        self.apperance_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, bias=True)
        self.apperance_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size,
                                        padding=(1, 1), bias=True)
        self.apperance_conv4 = nn.Conv2d(self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, bias=True)

        # Attention layers
        self.apperance_att_conv1 = nn.Conv2d(self.nb_filters1, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_1 = Attention_mask()
        self.apperance_att_conv2 = nn.Conv2d(self.nb_filters2, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_2 = Attention_mask()

        # Avg pooling
        self.avg_pooling_1 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_2 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_3 = nn.AvgPool2d(self.pool_size)

        # Dropout layers
        self.dropout_1 = nn.Dropout(self.dropout_rate1)
        self.dropout_2 = nn.Dropout(self.dropout_rate1)
        self.dropout_3 = nn.Dropout(self.dropout_rate1)
        self.dropout_4 = nn.Dropout(self.dropout_rate2)

        # Dense layers
        if img_size == 36:
            self.final_dense_1 = nn.Linear(3136, self.nb_dense, bias=True)
        elif img_size == 72:
            self.final_dense_1 = nn.Linear(16384, self.nb_dense, bias=True)
        elif img_size == 96:
            self.final_dense_1 = nn.Linear(30976, self.nb_dense, bias=True)
        else:
            raise Exception('Unsupported image size')

        # Final dense layer with a single neuron
        self.final_dense_2 = nn.Linear(self.nb_dense, 1, bias=True)
    def forward(self, inputs, params=None):
        diff_input = inputs[:, :3, :, :]
        raw_input = diff_input

        d1 = torch.tanh(self.motion_conv1(diff_input))
        d2 = torch.tanh(self.motion_conv2(d1))

        r1 = torch.tanh(self.apperance_conv1(raw_input))
        r2 = torch.tanh(self.apperance_conv2(r1))

        g1 = torch.sigmoid(self.apperance_att_conv1(r2))
        g1 = self.attn_mask_1(g1)
        gated1 = d2 * g1

        d3 = self.avg_pooling_1(gated1)
        d4 = self.dropout_1(d3)

        r3 = self.avg_pooling_2(r2)
        r4 = self.dropout_2(r3)

        d5 = torch.tanh(self.motion_conv3(d4))
        d6 = torch.tanh(self.motion_conv4(d5))

        r5 = torch.tanh(self.apperance_conv3(r4))
        r6 = torch.tanh(self.apperance_conv4(r5))

        g2 = torch.sigmoid(self.apperance_att_conv2(r6))
        g2 = self.attn_mask_2(g2)
        gated2 = d6 * g2

        d7 = self.avg_pooling_3(gated2)
        d8 = self.dropout_3(d7)
        d9 = d8.view(d8.size(0), -1)
        d10 = torch.tanh(self.final_dense_1(d9))
        d11 = self.dropout_4(d10)
        out = torch.sigmoid(self.final_dense_2(d11))  # Clone and detach to keep in the computation graph
        return out


class EnsembleModel(nn.Module):
    def __init__(self, model1, model2, model3, model4, weights):
        super(EnsembleModel, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.model4 = model4
        self.weights = weights

    def forward(self, x):
        preds1 = self.model1(x)
        preds2 = self.model2(x)
        preds3 = self.model3(x)
        preds4 = self.model4(x)

        # Apply softmax to convert logits to probabilities
        preds1_probs = torch.softmax(preds1, dim=1)
        preds2_probs = torch.softmax(preds2, dim=1)
        preds3_probs = torch.softmax(preds3, dim=1)
        preds4_probs = torch.softmax(preds4, dim=1)

        # Calculate the ensemble prediction (weighted average)
        weighted_predictions = (
            (preds1_probs * self.weights[0])
            + (preds2_probs * self.weights[1])
            + (preds3_probs * self.weights[2])
            + (preds4_probs * self.weights[3])
        )

        return weighted_predictions