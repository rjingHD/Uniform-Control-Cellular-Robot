#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
import os
import sys
import math
import time
import torch
import torch.nn.functional as F
import torch.optim as optim

from agent import Agent
from dqn_model import DQN

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)

PATH = "parameters"
MODEL = "DQNmodelgpu.pth"
BACKUPMODEL = "DQNmodelgpu_backup.pth"
#BACKUPPATH = "/home/rjing/code/python_ws/rl_ws/"
class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize everything 
            paramters for neural network  
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """
        super(Agent_DQN, self).__init__(env)
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.test_dqn = args.test_dqn
        self.env = env
        self.max_episodes = 100000
        self.eps_threshold = 1.0
        self.EPS_START = 1.0
        self.EPS_END = 0.025
        self.EPS_DECAY = 80000 #1200000
        self.EPS_SLOPE = (self.EPS_START - self.EPS_END)/self.EPS_DECAY
        self.gamma = 0.99
        self.target_update_period = 15
        self.starting_step = 5000
        self.training_frequency = 4

        self.img_w = args.img_width
        self.img_h = args.img_height
        self.img_d = args.img_channel
        self.learning_rate = args.learning_rate
        self.buffer_max_size = args.replay_buffer_size
        self.batch_size = args.batch_size
        self.load_model_training = args.load_model_training
        
        self.buffer_ptr = 0
        self.buffer_size = 0
        self.steps_done = 0
        
        self.last_episode_count = 0 
        self.episode_count = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Using device:', self.device)
        print()
        if self.device.type == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
        
        if not args.test_dqn:
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt
            self.timestr = time.strftime("%Y%m%d-%H:%M:%S")
            os.mkdir(PATH + "/" + self.timestr)
            self.path = PATH + "/" + self.timestr + "/" + MODEL
            self.backup_path = PATH + "/" + self.timestr + "/" + BACKUPMODEL
            
            self.state_buffer = np.zeros([self.buffer_max_size,self.img_d, self.img_h, self.img_w, ], dtype=np.uint8)
            self.next_state_buffer = np.zeros([self.buffer_max_size,self.img_d, self.img_h, self.img_w,], dtype=np.uint8)
            self.action_buffer = np.zeros([self.buffer_max_size], dtype=np.uint8)
            self.reward_buffer = np.zeros([self.buffer_max_size], dtype=np.int32)
            self.done_buffer = np.zeros([self.buffer_max_size], dtype=np.bool)
            
            self.policy_net = DQN(self.img_w, self.img_h, self.img_d).to(self.device)
            if self.load_model_training:
                print('loading previous model...')
                self.policy_net.load_state_dict(torch.load(MODEL))
                self.policy_net.eval()
            self.target_net = DQN(self.img_w, self.img_h, self.img_d).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
            self.optimizer = optim.Adam(self.policy_net.parameters(),lr=self.learning_rate)
        
        if args.test_dqn:
            self.policy_net = DQN(self.img_w, self.img_h, self.img_d).to(self.device)
            #you can load your model here
            print('loading trained model...')
            ###########################
            # YOUR IMPLEMENTATION HERE #
            self.policy_net.load_state_dict(torch.load(MODEL))
            self.policy_net.eval()

    # def init_game_setting(self):
    #     """
    #     Testing function will call this function at the begining of new game
    #     Put anything you want to initialize if necessary.
    #     If no parameters need to be initialized, you can leave it as blank.
    #     """

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        if test:
            with torch.no_grad():
                # observation = np.rollaxis(observation, 2)
                # observation = torch.from_numpy(observation).cuda()
                observation = torch.from_numpy(observation).cuda()
                observation = observation.view(-1,self.img_w*self.img_h).to(self.device)
                action = self.policy_net(observation.float()).max(1)[1].view(1, 1)
        else:  #train
            #self.eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
            if self.last_episode_count != self.episode_count:
                self.last_episode_count = self.episode_count
                self.eps_threshold = max(self.EPS_END,(self.eps_threshold-self.EPS_SLOPE))
            sample = random.random()
            self.steps_done += 1
            if sample > self.eps_threshold:
                with torch.no_grad():
                    #observation = np.rollaxis(observation, 2)
                    observation = torch.from_numpy(observation).cuda()
                    observation = observation.view(-1,self.img_w*self.img_h).to(self.device)
                    action = self.policy_net(observation.float()).max(1)[1].view(1, 1)
            else:
                action = torch.tensor([[random.randrange(2)]], device=self.device, dtype=torch.long)
        action_cpu = action.cpu().data.numpy()[0][0]
        return action_cpu

    def make_random_action(self):
        action = torch.tensor([[random.randrange(2)]], device=self.device, dtype=torch.long)
        action_cpu = action.cpu().data.numpy()[0][0]
        return action_cpu
    
    def push(self,
        state: np.ndarray,
        next_state: np.ndarray,
        action: float,
        reward: float,
        done: bool):
        """ can consider deque(maxlen = 10000) list
        """
        temp_state = state          #np.rollaxis(state,2)
        temp_next_state = next_state#np.rollaxis(next_state,2)
        self.state_buffer[self.buffer_ptr] = temp_state
        self.next_state_buffer[self.buffer_ptr] = temp_next_state
        self.action_buffer[self.buffer_ptr] = action
        self.reward_buffer[self.buffer_ptr] = reward
        self.done_buffer[self.buffer_ptr] = done
        self.buffer_ptr = (self.buffer_ptr + 1) % self.buffer_max_size
        self.buffer_size = min(self.buffer_size + 1, self.buffer_max_size)
    def replay_buffer(self):
        """ You can add additional arguments as you need.
        Select batch from buffer.
        """
        indexes = np.random.choice(self.buffer_size, size=self.batch_size, replace=False)
        return dict(state=self.state_buffer[indexes],
                    next_state=self.next_state_buffer[indexes],
                    action=self.action_buffer[indexes],
                    reward=self.reward_buffer[indexes],
                    done=self.done_buffer[indexes])

    def compute_dqn_loss(self, batch_samples: [str, np.ndarray]) -> torch.Tensor:
        if self.buffer_size < self.batch_size:
            return
        device = self.device  
        # state = torch.FloatTensor([np.rollaxis(s,2) for s in batch_samples["state"]]).to(device)
        # next_state = torch.FloatTensor([np.rollaxis(s,2) for s in batch_samples["next_state"]]).to(device)
        state = torch.FloatTensor(batch_samples["state"].reshape(-1,self.img_w*self.img_h)).to(device)
        next_state = torch.FloatTensor(batch_samples["next_state"].reshape(-1,self.img_w*self.img_h)).to(device)
        reward = torch.FloatTensor(batch_samples["reward"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(batch_samples["done"].reshape(-1, 1)).to(device)
        action = torch.LongTensor(batch_samples["action"]).to(device)
        
        #curr_q_value = torch.sum(self.policy_net(state)*action, dim=1).to(device)
        
        # curr_q_value = self.policy_net(state).gather(dim = 1, index = action.view(-1,1))
        # # # DQN
        # # # next_q_value = self.target_net(next_state).max(dim=1, keepdim=True)[0].detach()
        # # # Double DQN
        # next_q_value = self.target_net(next_state).gather(  # Double DQN
        #     dim = 1, index = self.policy_net(next_state).argmax(dim=1, keepdim=True)).detach()
        # mask = 1 - done
        # target = (reward + self.gamma*next_q_value*mask).to(self.device)
        
        curr_q_value = self.policy_net.forward(state).gather(1, action.view(-1,1)).squeeze()
        with torch.no_grad(): 
            next_q_value = self.target_net.forward(next_state).gather(  # Double DQN
                    1, torch.argmax(self.policy_net(next_state),dim=1, keepdim=True))
            mask = 1 - done
            target = (reward + self.gamma * next_q_value * mask).squeeze().to(self.device)

        # q_values      = self.policy_net(state)
        # next_q_values = self.policy_net(next_state)
        # next_q_state_values = self.target_net(next_state) 

        # q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1) 
        # next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        # expected_q_value = reward + self.gamma * next_q_value * (1 - done)
        #q_value - Variable(expected_q_value.data)
        # calculate dqn loss
        # print("current:",q_value)
        # print("target:",expected_q_value)
        #loss = F.smooth_l1_loss(q_value, expected_q_value)
        loss = F.smooth_l1_loss(curr_q_value, target)
        return loss

    def train(self):
        # Optimize the model
        if not self.test_dqn:
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt
        stepPermin = 50000
        fig=plt.figure()
        ax=fig.add_subplot(1,1,1)
        ax.set_xlabel('Number of training steps')
        ax.set_ylabel('Average reward over last 30 episodes')
        ax.set_title('Reward trajectory')
        line = ax.plot([0,0], [100000,200], '-b')[0]
        plt.grid(True)
        plt.ion()  #interactive mode on
        X = []
        Y = []
        Y_bar = []
        max_y_bar = 0
        start_update_flag = False
        self.episode_count = 0
        update_step = 0
        current_test_reward = self.test_policy()
        for episode in range(self.max_episodes):
            episode_reward = 0.0
            done = False
            observation,target_position = self.env.reset()
            while not done:
                if start_update_flag:
                    action = self.make_action(observation,test=False)
                else:
                    action = self.make_random_action()
                next_observation, reward, done, _ = self.env.step(action)
                episode_reward += reward
                self.push(observation,next_observation,action,reward,done)
                observation = next_observation

                #if self.buffer_ptr >= self.batch_size:
                    #  start_update_flag = True
                if (not start_update_flag) and (self.buffer_ptr >= self.starting_step):
                    #episode = 0
                    update_step = 0
                    self.eps_threshold = 1
                    start_update_flag = True
                    print("start training...")
                if start_update_flag:
                    if(update_step%(self.training_frequency)==0):
                        batch = self.replay_buffer()
                        loss = self.compute_dqn_loss(batch)
                        self.optimizer.zero_grad()
                        loss.backward()
                        # DQN
                        for param in self.policy_net.parameters():
                            param.grad.data.clamp_(-1, 1)
                        # END
                        # Dueling DQN
                        # # we clip the gradients to have their norm less than or equal to 10
                        # torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
                        # # we rescale the combined gradient entering the last convolutional layer by 1/√2 
                        # for name, m in self.policy_net.named_modules():
                        #     if (name == "conv3"):
                        #         for param in m.parameters():
                        #             param.grad *= 0.70710678
                        # END
                        self.optimizer.step()
                    #total_loss+=loss.cpu().detach().numpy()
                    if(update_step%(4*stepPermin) == 0):   #test every 4min training  
                        plt.savefig("parameters/"+self.timestr+"/experimental_result.png")
                        print("experiment plot saved.")
                    if(update_step%(0.1*stepPermin) == 0):  #update plot every 5s  
                        line.set_xdata(X)
                        line.set_ydata(Y_bar)
                        ax.set_xlim([0, self.episode_count+1])
                        ax.set_ylim([-1, max_y_bar+3])
                        plt.pause(0.001)
                        print('episode:',self.episode_count,'update_step:',update_step,'test_reward：',current_test_reward)
                    update_step += 1
                else:
                    if(self.buffer_ptr%1000 == 0):
                        print("preparing buffer：",self.buffer_ptr)
            if start_update_flag:
                self.episode_count += 1
                X.append(self.episode_count)
                Y.append(episode_reward)
                if self.episode_count<30:
                    y_bar = np.mean(Y)
                else: 
                    y_bar = np.mean(Y[-30:])
                    if y_bar - max_y_bar > 0.3:
                        max_y_bar = y_bar
                        torch.save(self.policy_net.state_dict(),self.path)
                        print("network weight saved. Best_reward:",max_y_bar)
                        if(self.episode_count > 500):
                            current_test_reward = self.test_policy()
                            print("current test reward:",current_test_reward)
                Y_bar.append(y_bar)
                if(self.episode_count%(self.target_update_period)==0):
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                if(self.episode_count%(2000)==1):
                    self.backup_path = PATH + "/" + self.timestr + "/" + "DQNmodelgpu_backup" + str(self.episode_count) +".pth"
                    torch.save(self.policy_net.state_dict(),self.backup_path)
                    print("network weight backup saved. Episode:",self.episode_count)
                if max_y_bar > 10000:
                    break
    def test_policy(self):
        print("Testing current policy...")
        rewards = []
        for i in range(30):
            #state = self.env.reset()
            observation,target_position = self.env.reset()
            done = False
            episode_reward = 0.0
            while(not done):
                action = self.make_action(observation, test=True)
                observation, reward, done, _ = self.env.step(action)
                episode_reward += reward
            rewards.append(episode_reward)
        current_test_reward = np.mean(rewards)
        print("Current policy reward:",current_test_reward)
        return current_test_reward