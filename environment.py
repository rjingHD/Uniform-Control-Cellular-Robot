"""

### NOTICE ###
DO NOT revise this file

"""
# import gym
import numpy as np
# from atari_wrapper import make_wrap_atari

class Environment(object):
    def __init__(self, env_name, args, test=False):
        # if atari_wrapper:
        #     clip_rewards = not test
        #     self.env = make_wrap_atari(env_name, clip_rewards)
        # else:

        # self.env = gym.make(env_name)

        # self.action_space = self.env.action_space
        # self.observation_space = self.env.observation_space
        self.env = CellEnv()
        
    def seed(self, seed):
        '''
        # Control the randomness of the environment
        # '''
        # self.env.seed(seed)

    def reset(self):
        '''
        When running dqn:
            observation: np.array
                stack 4 last frames, shape: (84, 84, 4)

        When running pg:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)
        '''
        observation,self.goal = self.env.reset()

        return np.array(observation),self.goal


    def step(self,action):
        '''
        When running dqn:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
            reward: int
                wrapper clips the reward to {-1, 0, 1} by its sign
                we don't clip the reward when testing
            done: bool
                whether reach the end of the episode?

        When running pg:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)
            reward: int
                if opponent wins, reward = +1 else -1
            done: bool
                whether reach the end of the episode?
        '''
        # if not self.env.action_space.contains(action):
        #     raise ValueError('Ivalid action!!')

        observation, reward, done = self.env.step(action)
        info = [""]

        return np.array(observation), reward, done, info


    # def get_action_space(self):
    #     return self.action_space


    # def get_observation_space(self):
    #     return self.observation_space


    # def get_random_action(self):
    #     return self.action_space.sample()

class CellEnv:
    grid_SIZE = 10.0
    Nb_robot = 10
    RETURN_IMAGES = False
    IMPROVE_REWARD = 1
    GOAL_REWARD = 30
    velocity = 0.1
    # ENEMY_PENALTY = 300 
    # FOOD_REWARD = 25
    # OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)  # 4
    #ACTION_SPACE_SIZE = 9
    
    # the dict! (colors)
    d = {1: (255, 175, 0),
         2: (0, 255, 0),
         3: (0, 0, 255)}
    
    def reset(self):
        self.reach_goal_flag = np.zeros((1, self.Nb_robot), dtype=bool)
        self.reach_goal_count = 0
        self.goal = goal(self.grid_SIZE)
        self.robots = []
        for i in range(self.Nb_robot):
            self.robots.append(robot(self.grid_SIZE,self.velocity))
            error = [self.goal.x-self.robots[i].x, self.goal.y-self.robots[i].y]
            while  np.linalg.norm(error, ord=2) < 0.1:
                self.robots[i] = robot(self.grid_SIZE,self.velocity)
                error = [self.goal.x-self.robots[i].x, self.goal.y-self.robots[i].y]
        self.episode_step = 0
        # if self.RETURN_IMAGES:
        #     #observation = np.array(self.get_image())
        # else:
        observation = []
        error = []
        self.target_state = [self.goal.x,self.goal.y,np.array([0.0])]
        for robot_i in self.robots:
            theta_reference_i = np.arctan2(self.goal.y-robot_i.y, self.goal.x-robot_i.x)
            error_i = [self.goal.x-robot_i.x, self.goal.y-robot_i.y, theta_reference_i - robot_i.theta]
            observation_i = [robot_i.x, robot_i.y, robot_i.theta]
            observation.append(observation_i) 
            error.append(error_i)
        observation.append(self.target_state)
        observation = np.concatenate(observation,axis=1)
        observation = observation.reshape(1,3,-1)
        error = np.concatenate(error,axis=1)
        error = error.reshape(1,3,-1)
        # print(type(observation))
        # print(observation.shape)
        self.last_observation = observation
        self.last_error = error
        return observation,self.goal
    
    def step(self, action):
        self.reach_goal_flag = np.zeros((1, self.Nb_robot), dtype=bool)
        self.episode_step += 1
        reward = 0
        # if self.RETURN_IMAGES:
        #     # new_observation = np.array(self.get_image())
        # else:
        for robot_i in self.robots:
            robot_i.follow_action(action)
        theta_reference = []
        error = []
        new_observation = []
        for robot_i in self.robots:
            theta_reference_i = np.arctan2(self.goal.y-robot_i.y, self.goal.x-robot_i.x)
            error_i = [self.goal.x-robot_i.x, self.goal.y-robot_i.y, theta_reference_i - robot_i.theta]
            observation_i = [robot_i.x, robot_i.y, robot_i.theta]
            theta_reference.append(theta_reference_i)
            new_observation.append(observation_i)
            error.append(error_i)
        new_observation.append(self.target_state)
        new_observation = np.concatenate(new_observation,axis=1)
        new_observation = new_observation.reshape(1,3,-1)
        error = np.concatenate(error,axis=1)
        error = error.reshape(1,3,-1)

        done = False
        
        i = 0
        for robot_i in self.robots:
            if abs(robot_i.x - self.goal.x)<3e-1 and abs(robot_i.y - self.goal.y)<3e-1:
                self.reach_goal_flag[0,i] = True
            i = i+1
        if sum(sum(np.abs(error[0,0:2,:])-np.abs(self.last_error[0,0:2,:]))) > 0.4:  
            reward += self.IMPROVE_REWARD
        if np.count_nonzero(self.reach_goal_flag)>self.reach_goal_count:
            self.reach_goal_count=np.count_nonzero(self.reach_goal_flag)
            reward += self.GOAL_REWARD
        if  self.episode_step >= 200:  #reward == self.FOOD_REWARD or reward == -self.ENEMY_PENALTY or
            done = True
        self.last_observation = new_observation[:][:][:]
        self.last_error = error[:][:][:]
        return new_observation, reward, done

    # def render(self):
        # img = self.get_image()
        # img = img.resize((300, 300))  # resizing so we can see our robot in all its glory.
        # cv2.imshow("image", np.array(img))  # show it!
        # cv2.waitKey(1)

class goal:
    def __init__(self, size):
        self.size = size
        self.x = np.random.uniform(0.1*size, 0.9*size, 1) # dont be too close to the boundary
        self.y = np.random.uniform(0.1*size, 0.9*size, 1) ##np.random.random(0.1*size, 0.9*size)

class robot:
    def __init__(self, size,velocity):
        self.size = size
        self.velocity = velocity
        self.x = np.random.uniform(0.05*size, 0.95*size, 1) 
        self.y = np.random.uniform(0.05*size, 0.95*size, 1) 
        self.theta = np.random.uniform(-np.pi,np.pi,1)
        self.mu, self.sigma =  np.pi/4, np.pi/180*5 # mean and standard deviation

    def __str__(self):
        return f"mi-robot ({self.x}, {self.y},{self.theta})"

    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y, self.theta-other.theta)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def follow_action(self, choice):
        if choice == 1:
            delta_theta = np.random.normal(self.mu, self.sigma)
            self.theta = self.theta + delta_theta
            if self.theta > np.pi:
                self.theta -= 2*np.pi
            elif self.theta < -np.pi:
                self.theta += 2*np.pi
        elif choice == 0:
            self.theta = self.theta
        self.move(xx=self.velocity*np.cos(self.theta), yy=self.velocity*np.sin(self.theta))

    def move(self, xx=False, yy=False):

        # If no value for x, move randomly
        if not xx:
            print("no x")# self.x += np.random.randint(-1, 2)
        else:
            self.x += xx

        # If no value for y, move randomly
        if not yy:
            print("no y")# self.y += np.random.randint(-1, 2)
        else:
            self.y += yy

        # If we are out of bounds, fix!
        if self.x < 0:
            self.x = np.array([0.])
            #print("xlow")
        elif self.x > self.size:
            self.x = np.array([self.size])
            #print("xhigh")
        if self.y < 0:
            self.y = np.array([0.])
            #print("ylow")
        elif self.y > self.size:
            self.y = np.array([self.size])
            #print("yhigh")