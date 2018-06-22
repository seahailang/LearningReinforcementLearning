#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm
@file: snake_env.py
@time: 2018/6/21 15:45
"""

import numpy as np
import gym
from gym.spaces import Discrete

class SnakeEnv(gym.Env):
    SIZE=100

    def __init__(self,ladder_num,dices):
        # 初始化话蛇棋游戏
        # ladder_num 表示有多少个梯子，梯子会在后面的代码中随机分配
        # dices 表示有多少种投掷色字的策略
        # SIZE规定了整个棋盘有多大

        self.ladder_num = ladder_num
        self.dices = dices
        self.ladders = dict(np.random.randint(1,self.SIZE,size=(self.ladder_num,2)))
        self.observation_space = Discrete(self.SIZE+1)
        self.action_space = Discrete(len(dices))

        items = list(self.ladders.items())
        #双向的梯子
        for k,v in items:
            self.ladders[v] = k
        print('ladders info:')
        print(self.ladders)
        print('dice ranges')
        print(self.dices)

        self.pos = 1

    def _reset(self):
        self.pos=1
        return self.pos

    def _step(self,a):
        step = np.random.randint(1,self.dices[a]+1)
        self.pos += step
        if self.pos ==100:
            return 100,100,1,{}
        elif self.pos >100:
            self.pos = 200 -self.pos
        if self.pos in self.ladders:
            self.pos = self.ladders[self.pos]
        return self.pos,-1,0,{}

    def reward(self,s):
        if s == 100:
            return 100
        else:
            return -1

if __name__ == '__main__':
    s = SnakeEnv(10,[10])