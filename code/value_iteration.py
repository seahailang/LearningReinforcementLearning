#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm
@file: value_iteration.py
@time: 2018/6/22 14:04
"""
from snake_env import SnakeEnv
from agent import TableAgent
from numpy import np

class ValueIteration(object):
    def value_iteration(self,agent,max_iterator=-1):
        iteration = 0
        while True:
            iteration += 1
            new_value_pi = np.zeros_like(agent.value_pi)
            value_sas = []
            for i in range(1,agent.s_len):
                for j in range(agent.a_len):
                    value_sa = np.dot(agent.p[j,i,:],agent.r+agent.gamma*agent.value_pi)
                    value_sas.append(value_sa)
            new_value_pi = max(value_sas)
            diff = np.sqrt(np.sum(np.square(agent.value_pi-new_value_pi)))
            if diff <1e-6:
                break
            else:
                agent.value_pi = new_value_pi
            if iteration == max_iterator:
                break

        print('Iteration converged at round %d'%iteration)

        for i in range(1,agent.s_len):
            for j in range(0,agent.a_len):
                agent.value_q[i,j] = np.dot(agent.p[i,j],agent.r+agent.gamma*agent.value_pi)
            max_act = np.argmax(agent.value_q[i,:])
            agent.pi[i] = max_act



if __name__ == '__main__':
    pass