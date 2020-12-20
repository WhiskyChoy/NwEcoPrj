  
# -*- coding: utf-8 -*-
"""MyProblem.py"""
import numpy as np
import geatpy as ea
import math

class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, n, T, ep, df_c_max, df_d, df_s):
        self.n = n
        self.range_index = range(0, n)
        self.T = T
        self.ep = ep
        self.df_d = df_d
        self.df_s = df_s

        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        M = 1  # 初始化M（目标维数）
        maxormins = [-1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = n * (n-1)  # 初始化Dim（决策变量维数）
        varTypes = [1] * Dim  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [0] * Dim # 决策变量下界
        ub = [df_c_max[i][j]
             for i in self.range_index for j in self.range_index if i != j]  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def index_trans(self, i, j):
        if i > j:
            return i*self.n+j-i
        if i < j:
            return i*self.n+j-i-1

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        d = [self.df_d[i][j]
         for i in self.range_index for j in self.range_index if i != j]
        pop.ObjV = np.sum(Vars * d, 1, keepdims=True)        
        # 采用可行性法则处理约束
        pop.CV = np.hstack([ (self.df_s[i] * Vars[:,[self.index_trans(i,j)]] - self.df_s[j] *
                Vars[:,[self.index_trans(j,i)]] + self.T * math.log((1-self.ep)/self.ep)) for i in self.range_index
            for j in self.range_index
            if i != j])

