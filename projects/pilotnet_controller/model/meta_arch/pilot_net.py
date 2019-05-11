
import torch
import torch.nn as nn
from model.layer.feed_forward import FeedForward


class PilotNet(nn.Module):
    def __init__(self, cfg, visualizing=False):
        super(PilotNet, self).__init__()
        self.cfg = cfg
        self.visualizing = visualizing

        # BUILD CNN BACKBONE
        cnn_layers = []
        input_channels = self.cfg.MODEL.CNN.INPUT_CHANNELS
        cnn_configs = self.cfg.MODEL.CNN.LAYERS
        for cnn_config in cnn_configs:
            cnn_layer = [nn.Conv2d(input_channels,
                                   cnn_config['out_channels'],
                                   cnn_config['kernel'],
                                   cnn_config['stride']),
                         nn.ELU(),
                         nn.Dropout2d(p=self.cfg.MODEL.CNN.DROPOUT)]
            input_channels = cnn_config['out_channels']
            cnn_layers.extend(cnn_layer)

        self.cnn_backbone = nn.Sequential(*cnn_layers)

        # BUILD FULLY CONNECTED
        self.embedding = FeedForward(self.cfg)
        last_embedding_size = self.cfg.MODEL.FC.LAYERS[-1]['to_size']
        self.to_out = nn.Linear(last_embedding_size, 1)
        self.feed_forward = nn.Sequential(self.embedding, self.to_out)

        # BUILD LOSS CRITERION
        self.loss_criterion = nn.MSELoss()

    def forward(self, input, targets=None):
        batch_size = input.size(0)
        # if image pixel range is 0-255 use 127.5 as opposed to 0.5
        assert(input.max() <= 1.0)
        normalized_input = input / 0.5 - 1
        cnn_features = self.cnn_backbone(normalized_input)
        flattened_features = cnn_features.view([batch_size, -1])
        #print(flattened_features.size())
        predictions = self.feed_forward(flattened_features)

        if self.visualizing:
            activations = []
            layers_activation = normalized_input
            for i, module in enumerate(self.cnn_backbone.children()):
                layers_activation = module(layers_activation)
                if type(module) == nn.ELU:
                    layers_activation_temp = layers_activation.clone()
                    layers_activation_temp = layers_activation_temp.detach()
                    layers_activation_temp = layers_activation_temp.mean(1, keepdim=True)
                    activations.append(layers_activation_temp)
            return predictions, activations

        return predictions
