#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Component: YLPF
# Package  : battm
# Name     : UKF
# Author   : NBIC
# License  : BSD 2-Clause
# -------------------------------------------------------------------
#
# Copyright (C) 2019 NEC Brain Inspired Computing
#                    Research Alliance Laboratories
#                    All Rights Reserved.
# -------------------------------------------------------------------
#
# -------------------------------------------------------------------
# Provider: NEC Brain Inspired Computing Research Alliance Laboratories
#      DOI: https://doi.org/10.1371/journal.pcbi.1004442
#    Title: A Bayesian Attractor Model for Perceptual Decision Making
#    Class: UKF
#     Role: UKF Class.
# -------------------------------------------------------------------
# 
import math
import numpy as np

from hopfield import tanh_sigmoid

# unscented kalman filter(UKF)
# UKF の論文: https://doi.org/10.1109/ASSPCC.2000.882463

# 予測結果
class PredictedResult(object):
    """ 予測結果 """
    def __init(self):
        self.predx = None 
        self.predP = None
        self.samples = None


# 推定結果
class EstimationResult(object):
    """ 推定結果 """
    def __init__(self):
        self.estx = None
        self.estP = None
        
# system model
# x' = f(x,w)  ; n-vector
# y = h(x) + v ; m-vector
#
# given:
#  E[w]=0, V[w]=Q
#  E[v]=0, V[v]=R
#  initial state: x0|0, P0|0
class UKFModel(object):
    def __init__(self, n, m, J, obs, Q, R, x0, P0):
        self.n = n
        self.m = m
        # Python ではラムダ関数ではなく行列 J を渡す
        self.J = J
        # Pytyon ではラムダ関数 h ではなく、要素リスト obs を渡す
        self.obs = obs
        self.Q = Q
        self.R = R
        self.x0 = x0
        self.P0 = P0
        pass

def UKF_makeModel(n, m, J, obs, Q, R, x0, P0):
    model = UKFModel(n, m, J, obs, Q, R, x0, P0)
    return model

# xe : x_k-1|k-1
# Pe : P_k-1|k-1
def UKF_predict(cfg, xe, Pe):
    model = cfg.BAttM_model
    logger = model._logger

    num_w = int(math.sqrt(model.Q.size))
    #  #  estx：1×nの行列。
    #  # 意思決定状態 z の期待値 (est[[t]]$estx)
    #  # 各カテゴリのzの期待値が格納されている。
    estX = np.hstack((xe, np.zeros(num_w)))

    #  # estP：n×xの共分散行列。
    #  # 共分散行列：est[[t]]$estP
    #  # カテゴリ間の共分散が格納されている。
    estP = np.hstack((Pe, np.zeros((num_w, model.n))))
    estP = np.vstack((estP, np.hstack((np.zeros((model.n, num_w)), model.Q))))
    
    # sampling
    # sigma点を計算する
    samples_X = UKF_sample(cfg, estX, estP, (model.n + num_w))

    weights_s = UKF_weights_s(cfg, (model.n + num_w))
    weights_c = UKF_weights_c(cfg, (model.n + num_w))

    # U-transformation
    samples_x2 = UKF_Utransformation_f(cfg, samples_X)

    predx = list_weightedSum(samples_x2, weights_s)
    p = [*map(lambda x: x - predx, samples_x2)]
    predP = list_weightedSquareSum(cfg, p, weights_c)

    res = PredictedResult()
    res.predx = predx
    res.predP = predP
    res.samples = samples_x2
    return res

#UKF.update = function(model,pred,y){
def UKF_update(cfg, pred, y):
    model = cfg.BAttM_model
    logger = model._logger

    num_v = int(math.sqrt(model.R.size))
    num_w = int(math.sqrt(model.Q.size))
    num_x = int(pred.predx.size)
    logger.debug("UKF Update v:{} w:{} x:{}".format(num_v, num_w, num_x))

    #  #sampling
    samples_X = pred.samples
    weights_s = UKF_weights_s(cfg, (model.n + num_w))
    weights_c = UKF_weights_c(cfg, (model.n + num_w))

    #  # U-transformation 
    #  # Sigma点列の広がり具合を決めるために Unscented 変換する
    samples_y = UKF_Utransformation_h(samples_X, model.obs)

    predy = list_weightedSum(samples_y, weights_s)
    pyy = [*map(lambda x: x - predy, samples_y)]
    wpyy = list_weightedSquareSum(cfg, pyy, weights_c)
    predPyy = wpyy + model.R
    pxy_l1 = [*map(lambda x: x - pred.predx, pred.samples)]
    pxy_l2 = [*map(lambda x: x - predy, samples_y)]
    predPxy = list_weightedCrossSum(cfg, pxy_l1, pxy_l2, weights_c)

    K = np.dot(predPxy, np.linalg.inv(predPyy))

    estx = pred.predx + np.dot(K, (y - predy))
    estP = pred.predP - np.dot(np.dot(K, predPyy), K.T)

    res = EstimationResult()
    res.estx = estx
    res.estP = estP
    return res


# Sigma点列を計算する
def UKF_sample(Cfg, estX, estP, n):
    logger = Cfg.BAttM_model._logger

    x = list()

    estSigma = matrix_sqrt(estP)
    #  # UKFの論文 p.3 左中央の式
    #  # \lambda = alpha^2 (L + \kappa) - Lに従う
    #  lambda = Cfg$UKF.alpha^2 * (n + Cfg$UKF.kappa) - n
    lambd = math.pow(Cfg.UKF.alpha, 2) * (n + Cfg.UKF.kappa) - n

    #  # UKFの論文 p.3 左中央の式(15)の上から三つに従う
    x.append(estX)

    for i in np.arange(0, n):
        x.append(estX + math.sqrt(n + lambd) * estSigma[i, ])
        x.append(estX - math.sqrt(n + lambd) * estSigma[i, ])
    logger.debug("lambd:{}".format(lambd))
    return x


# 関数で Unscented 変換を行う
def UKF_Utransformation_f(Cfg, samples):
    model = Cfg.BAttM_model
    logger = model._logger
    ukf = model.UKF
    n = model.n

    res = list()
    # リストの各要素に対して関数を適用する
    for i in np.arange(0, len(samples)):
        # UKF論文のUnscented変換で使用される関数
        # Ff = function(zw){
        #   zw[1:n] + f(zw[1:n]) + zw[n + 1:n]
        # }
        # Ff = lambda zw: zw[1:n] + f(zw[1:n]) + zw[n + 1:n]
        zw = samples[i]
        f = ff_sigmoid(zw[:n], ukf.J, model.tau, model.rambda)
        ff = zw[:n] + f + zw[n:]
        res.append(ff)
    return np.array(res)


# f = lambda z: ((-1/tau) * z) + (J * tanh(rambda * z))
def ff_sigmoid(x, J, tau, rambda):
    res = ((-1.0/tau) * x) + np.dot(J, np.tanh(rambda * x))
    return res


# 観測関数 obs (h) 用の UKF Unscented 変換関数
def UKF_Utransformation_h(samples_X, obs):
    obs_list   = obs[0]
    state_list = obs[1]
    r          = obs[2]
    o          = obs[3]
    
    result = list()
    # obs 関数の定義 (UKF で使用される)
    n = len(obs_list)

    for c in np.arange(0, len(samples_X)):
        ans = 0
        z = samples_X[c]
        #  hopfield.R で定義される sigmoid 関数を参照する
        res_sig = tanh_sigmoid(z, r, o)

        #  # すべてのアトラクタに対して
        for i in np.arange(0, n):
            # 大歳先生論文の式(5)
            list_1 = state_list[i] == 1
            list_m1 = state_list[i] == -1
            idx = np.prod(res_sig[list_1]) * np.prod(1 - res_sig[list_m1])
            ans += obs_list[i] * idx
        result.append(ans)

    return result


# Sigma点列用のウェイト W^{(m)}_0 を計算する
def UKF_weights_s(Cfg, n):
    lambd = math.pow(Cfg.UKF.alpha, 2) * (n + Cfg.UKF.kappa) - n
    ws0 = lambd / (n + lambd)
    ws = list()
    ws.append(ws0)
    
    for i in np.arange(0, n):
        ws_p = 1.0 / (2.0 * (n + lambd))
        ws.append(ws_p)
        ws_n = 1.0 / (2.0 * (n + lambd))
        ws.append(ws_n)
    return ws


# Sigma点列用のウェイト W^{(c)}_0 を計算する
def UKF_weights_c(Cfg, n):
    # lambda = Cfg$UKF.alpha^2 * (n + Cfg$UKF.kappa) - n
    lambd = math.pow(Cfg.UKF.alpha, 2) * (n + Cfg.UKF.kappa) - n
    # # W^{(c)}_0 = \lambda / (L + \lambda) + (1 - \alpha^2 + \beta)
    ws0 = lambd / (n + lambd) + (1 - math.pow(Cfg.UKF.alpha, 2) + Cfg.UKF.beta)
    
    ws = list()
    ws.append(ws0)
    
    for i in np.arange(0, n):
        ws_p = 1.0 / (2.0 * (n + lambd))
        ws.append(ws_p)
        ws_m = 1.0 / (2.0 * (n + lambd))
        ws.append(ws_m)
    return ws


# 行列の平方根を求める
#matrix.sqrt = function(A){
#  Dbg.fbegin(match.call()[[1]])
#  
#  # 特異値分解(SVD)を行う
#  # X = U %*% D %*% V を満たす U, D, V を求める
#  tmp = svd(A)
#  
#  # 直行行列 U
#  U = tmp$u
#  
#  # 直行行列 V
#  V = tmp$v
#  
#  # X の特異値を対角成分とする対角行列 D
#  # 単位行列を作成する
#  D = diag(sqrt(tmp$d))
#  
#  Dbg.fend(match.call()[[1]])
#  return( U %*% D %*% t(V))
#}
def matrix_sqrt(A):
    U, d, V =  np.linalg.svd(A, full_matrices=True)
    D = np.diag(np.sqrt(d))
    ms = np.dot(U, D)
    ms = np.dot(ms, V)
    return ms


# リストの重み付き和を計算する
# ans = \sum^{n}_{i=1} l_i \times w_i
def list_weightedSum(l, weights):
    ans = 0
    for i in np.arange(0, len(l)):
        ans += l[i] * weights[i]
    return ans


# リストの重み付き平方和をもとめる
# ans = \sum^{n}_{i=1} l_i * {}^t l_i \times w_i
def list_weightedSquareSum(cfg, l, weights):
    ans = None
    
    for i in np.arange(0, len(l)):
        outer_dot = np.outer(l[i], l[i].reshape(-1, 1))
        res = weights[i] * outer_dot
        if ans is None:
            ans = res
        else:
            ans += res

    return ans


# リストの重み付き直交和をもとめる
# ans = \sum^n_{i=1} (l1_i * {}^t l2_i ) \times w_i
def list_weightedCrossSum(cfg, l1, l2, weights):
    ans = None

    for i in np.arange(0, len(l1)):
        res = weights[i] * np.outer(l1[i], l2[i].reshape(-1, 1))
        if ans is None:
            ans = res
        else:
            ans += res

    return ans


