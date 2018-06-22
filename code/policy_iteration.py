#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm
@file: policy_iteration.py
@time: 2018/6/21 16:26
"""
from snake_env import SnakeEnv
from agent import TableAgent
import numpy as np

def eval_game(env,policy):
    state = env.reset()
    return_val = 0
    while True:
        if isinstance(policy,TableAgent):
            act = policy.play(state)
        elif isinstance(policy,list):
            act = policy[state]
        else:
            raise ValueError('Illegal policy')
        state,reward,terminate,_ = env.step(act)
        return_val += reward
        if terminate:
            break
    return return_val


class PolicyIteration(object):
    def policy_evaluation(self,agent,max_iter=-1):
        iteration = 0
        while True:
            iteration += 1
            new_value_pi = agent.value_pi.copy()  #v(s)
            for i in range(1,agent.s_len):
                ac = agent.pi[i] # act follow by policy
                # for j in range(0,agent.act_num): # for every act
                transition = agent.p[ac,i,:] #compute this p(s_(t+1)|s,a)
                value_sa = np.dot(transition,agent.r+agent.gamma*agent.value_pi) # q(s,a) = p(s_(t+1)|s,a)[r+q(s_(t+1),a)]
                    # value_sas.append(value_sa)
                new_value_pi[i] = value_sa # v(s) = pi(s)*q(s,a),because this policy only make one action

            diff = np.sqrt(np.sum(np.square(agent.value_pi-new_value_pi)))
            if diff>1e-6:
                break
            else:
                agent.value_pi = new_value_pi
            if iteration == max_iter:
                break

    def policy_improvement(self,agent):
        new_policy = np.zeros_like(agent.pi)
        for i in range(1,agent.s_len):
            for j in range(0,agent.a_len):
                agent.value_q[i,j] = np.dot(agent.p[i,j,:],agent.r+agent.gamma*agent.value_pi)
                max_act = np.argmax(agent.value_q[i,:])
                new_policy[i] = max_act
        if np.all(np.equal(new_policy,agent.pi)):
            return False
        else:
            agent.pi = new_policy
            return True

    def policy_iteration(self,agent):
        iteration = 0
        while True:
            iteration += 1
            self.policy_evaluation(agent)
            ret = self.policy_improvement(agent)
            if not ret:
                break
        print('Iteration end at round %i'%iteration)









def test_easy():
    sum_opt = 0
    sum_0 = 0
    sum_1 = 0
    env = SnakeEnv(0,[3,6])

    policy_ref = [1]*97+[0]*3
    policy_0 = [0]*100
    policy_1 = [1]*100

    for i in range(1000):
        sum_opt += eval_game(env,policy_ref)
        sum_0 += eval_game(env,policy_0)
        sum_1 += eval_game(env,policy_1)
    print('opt %f'%(sum_opt/1000))
    print('0 %f'%(sum_0/1000))
    print('1 %f'%(sum_1/1000))

if __name__ == '__main__':
    pass