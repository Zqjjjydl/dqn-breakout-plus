from typing import (
    Tuple,
)

import torch
import numpy as np
from copy import deepcopy

from utils_types import (
    BatchAction,
    BatchDone,
    BatchNext,
    BatchReward,
    BatchState,
    TensorStack5,
    TorchDevice,
)
# class One_transaction:
#     def __init__(self,channels,capacity,states,actions,rewards,dones):
#         self.__m_states = torch.zeros(
#             (capacity, channels, 84, 84), dtype=torch.uint8)
#         self.__m_actions = torch.zeros((capacity, 1), dtype=torch.long)
#         self.__m_rewards = torch.zeros((capacity, 1), dtype=torch.int8)
#         self.__m_dones = torch.zeros((capacity, 1), dtype=torch.bool)
#         self.__m_states = deepcopy(states)
#         self.__m_action = deepcopy(actions)
#         self.__m_reward = deepcopy(rewards)
#         self.__m_dones = deepcopy(dones)

class SumTree(object):
    data_pointer = 0

    def __init__(self, capacity,channels):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        
        self.__m_states = torch.zeros(
            (capacity, channels, 84, 84), dtype=torch.uint8)
        self.__m_actions = torch.zeros((capacity, 1), dtype=torch.long)
        self.__m_rewards = torch.zeros((capacity, 1), dtype=torch.int8)
        self.__m_dones = torch.zeros((capacity, 1), dtype=torch.bool)

    def add(self, p, folded_state,action,reward,done):
        tree_idx = self.data_pointer + self.capacity - 1
        # self.data[self.data_pointer] = data  # update data_frame
        self.__m_states[self.data_pointer] = folded_state
        self.__m_actions[self.data_pointer, 0] = action
        self.__m_rewards[self.data_pointer, 0] = reward
        self.__m_dones[self.data_pointer, 0] = done
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.__m_states[data_idx],\
            self.__m_actions[data_idx, 0],self.__m_rewards[data_idx, 0],\
                self.__m_dones[data_idx, 0]

    @property
    def total_p(self):
        return self.tree[0]  # the root

class ReplayMemory(object):

    def __init__(
            self,
            channels: int,
            capacity: int,
            device: TorchDevice,
    ) -> None:
        self.__device = device
        self.__capacity = capacity
        self.__size = 0
        self.__pos = 0

        self.__m_states = torch.zeros(
            (capacity, channels, 84, 84), dtype=torch.uint8)
        self.__m_actions = torch.zeros((capacity, 1), dtype=torch.long)
        self.__m_rewards = torch.zeros((capacity, 1), dtype=torch.int8)
        self.__m_dones = torch.zeros((capacity, 1), dtype=torch.bool)

    def push(
            self,
            folded_state: TensorStack5,
            action: int,
            reward: int,
            done: bool,
    ) -> None:
        self.__m_states[self.__pos] = folded_state
        self.__m_actions[self.__pos, 0] = action
        self.__m_rewards[self.__pos, 0] = reward
        self.__m_dones[self.__pos, 0] = done

        self.__pos = (self.__pos + 1) % self.__capacity
        self.__size = max(self.__size, self.__pos)

    def sample(self, batch_size: int) -> Tuple[
            BatchState,
            BatchAction,
            BatchReward,
            BatchNext,
            BatchDone,
    ]:
        indices = torch.randint(0, high=self.__size, size=(batch_size,))
        b_state = self.__m_states[indices, :4].to(self.__device).float()
        b_next = self.__m_states[indices, 1:].to(self.__device).float()
        b_action = self.__m_actions[indices].to(self.__device)
        b_reward = self.__m_rewards[indices].to(self.__device).float()
        b_done = self.__m_dones[indices].to(self.__device).float()
        return b_state, b_action, b_reward, b_next, b_done

    def __len__(self) -> int:
        return self.__size


class PriReplayMemory(object):
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error
    def __init__(
            self,
            channels: int,
            capacity: int,
            device: TorchDevice,
    ) -> None:
        self.__device = device
        self.__capacity = capacity
        self.__channels=channels
        self.__size = 0#没啥用
        self.__pos=0#没啥用 

        self.tree=SumTree(capacity,channels)


    def push(
            self,
            folded_state: TensorStack5,
            action: int,
            reward: int,
            done: bool,
    ) -> None:
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p,folded_state,action,reward,done)   # set the max p for new p

        #没啥用，但是懒得改了
        self.__pos = (self.__pos + 1) % self.__capacity
        self.__size = max(self.__size, self.__pos)

    def sample(self, batch_size: int) -> Tuple[
            BatchState,
            BatchAction,
            BatchReward,
            BatchNext,
            BatchDone,
    ]:
        b_idx =torch.zeros((batch_size,), dtype=torch.int32)
        ISWeights=torch.zeros((batch_size, 1))
        # b_state=np.empty((batch_size, self.tree.data[0].size))
        b_states=torch.zeros((batch_size, self.__channels, 84, 84), dtype=torch.uint8)
        b_actions = torch.zeros((batch_size, 1), dtype=torch.long)
        b_rewards = torch.zeros((batch_size, 1), dtype=torch.int8)
        b_dones = torch.zeros((batch_size, 1), dtype=torch.bool)

        pri_seg = self.tree.total_p / batch_size       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
        for i in range(batch_size):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, state,action,reward,done = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            if min_prob==0:
                min_prob=10**(-4)
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i], b_states[i, :], b_actions[i, :], b_rewards[i, :], b_dones[i, :] = idx, state,action,reward,done
        # return b_idx, b_memory, ISWeights

        return b_idx.to(self.__device), ISWeights.to(self.__device),b_states[:,:4].to(self.__device).float(), b_actions.to(self.__device), \
            b_rewards.to(self.__device).float(),b_states[:,1:].to(self.__device).float(), b_dones.to(self.__device).float()

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

    def __len__(self) -> int:
        return self.__size