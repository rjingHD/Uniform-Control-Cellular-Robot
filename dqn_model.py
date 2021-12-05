#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, width, height, in_channels=1, num_actions=2):
        """
        Parameters:
        -----------
        in_channels: number of channel of input.
        """
        super(DQN, self).__init__()
        self.f1 = nn.Linear(width*height, 128)
        self.f2 = nn.Linear(128, 256)
        self.f3 = nn.Linear(256, 2)

        # self.layers = nn.Sequential(
        #     nn.Linear(3*10, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 2)
        # )


    def forward(self, x):
        
        x = self.f1(x)
        x = F.relu(x)
        x = self.f2(x)
        x = F.relu(x)
        x = self.f3(x)
        x = F.softmax(x,dim=1)
        return x
        #return self.layers(x)
        
        ###########################


        # self.num_actions = num_actions
        # self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        # self.bn1 = nn.BatchNorm2d(32)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # self.bn2 = nn.BatchNorm2d(64)
        # self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # self.bn3 = nn.BatchNorm2d(64)
        # def conv2d_size_out(size, kernel_size = 5, stride = 2):
        #     return (size - (kernel_size - 1) - 1) // stride  + 1
        # conv_width = conv2d_size_out(size = conv2d_size_out(size = conv2d_size_out(
        #     size = width, kernel_size=8, stride=4),kernel_size=4,stride=2),kernel_size=3,stride=1)
        # conv_height = conv2d_size_out(size = conv2d_size_out(size = conv2d_size_out(
        #      size = height, kernel_size=8, stride=4),kernel_size=4,stride=2),kernel_size=3,stride=1)
        # linear_input_size = conv_width * conv_height * 64
        # self.fc1_adv = nn.Linear(in_features=linear_input_size, out_features=512)
        # self.fc1_val = nn.Linear(in_features=linear_input_size, out_features=512)
        # self.fc2_adv = nn.Linear(in_features=512, out_features=self.num_actions)
        # self.fc2_val = nn.Linear(in_features=512, out_features=1)

         ###########################
        # YOUR IMPLEMENTATION HERE #
        # x = F.relu(self.bn1(self.conv1(x)))
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))
        # x = x.view(x.size(0), -1) 
        # #x = F.relu(self.ln1(x.view(x.size(0), -1)))
        
        # adv = F.relu(self.fc1_adv(x))
        # val = F.relu(self.fc1_val(x))

        # adv = self.fc2_adv(adv)
        # val = self.fc2_val(val).expand(x.size(0), self.num_actions)
        
        # #x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.num_actions)
        # x = val + adv - adv.mean(dim=-1, keepdim=True)
        # return x