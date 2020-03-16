#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Component: YLPF
# Package  : battm
# Name     : BAttMModel
# Author   : NBIC
# License  : BSD 2-Clause
# -------------------------------------------------------------------
# -*- coding: utf-8 -*-
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
#    Class: BAttM for traffic simulator.
#     Role: BAttM class.
# -------------------------------------------------------------------
# 
# 
import math
import numpy as np
from scipy.stats import multivariate_normal

import UKF

#BAttM for time series
#
# * making hopfield function f with given time series
# * remenber the time series with observed value

class BAttMModel(object):
    def __init__(self, logger, att_list, rambda, obs, q, s, z0, P0):
        self._logger = logger
        # # カテゴリ数 + 1 となるアトラクタ次元数
        self.n = len(att_list[0])

        # # 特徴量の次元数
        self.m = obs[0].shape[1]
        self._logger.debug("BAttMModel n:{} m:{}".format(self.n, self.m))

        # # 随時更新されるホップフィールド関数
        # # 論文の式(1)及び式(2)に登場する f(z)
        # f = gen.hopfield_PI(att_list,1,1,rambda)
        self.J = None
        self.tau = None
        self.rambda = None
        # UKF論文のUnscented変換で使用される関数
        # Ff = function(zw){
        #   zw[1:n] + f(zw[1:n]) + zw[n + 1:n]
        # }
        # Python では定義しない

        # # 随時更新される観測関数
        # h = function(z){
        #   obs(z)
        # }
        # Python では定義しない
        # h = lambda z: obs(z)
        self.obs = obs

        # # Q：n×nの対角行列、
        # # BAttM.q の二乗で計算される動的な不確実性。
        # # 論文 式(2)の直下の Q
        # # カテゴリ数=2の場合：
        # # [[0.24, 0.00, 0.00],
        # #  [0.00, 0.24, 0.00],
        # #  [0.00, 0.00, 0.24]]
        # Q = diag(q^2, n)
        self.Q = np.diag(np.ones(self.n)) * math.pow(q, 2)
        # # R：m×mの対角行列、
        # # s の二乗で計算されるセンサー誤差（分散）
        # # 論文の p.7 にある $\vec{R} = r^2$ のこと。
        # # 特徴量次元数=3の場合：
        # # [[0.01, 0.00, 0.00],
        # #  [0.00, 0.01, 0.00],
        # #  [0.00, 0.00, 0.01]]
        # R = diag(s^2, m)
        self.R = np.diag(np.ones(self.m)) * math.pow(s, 2)
        self._logger.debug("R:{}".format(self.R.shape))

        # if(is.null(z0)){
        #   z0 = rep(0,n)
        # }
        if z0 is None:
            self.z0 = np.zeros(self.n)
        else:
            self.z0 = z0

        # if(is.null(P0)){
        #   # P0：n×nの対角行列
        #   P0 = diag(1,n)
        # }
        if P0 is None:
            self.P0 = np.diag(np.ones(self.n))
        else:
            self.P0 = P0
        # Dbg.fend(match.call()[[1]])
        # UKF.makeModel(n,m,Ff,h,Q,R,z0,P0)
        # self.UKF = self.makeUKFModel(n,m,Ff,h,Q,R,z0,P0)
        self.UKF = None
        # }
        pass


class BAttM(object):
    def __init__(self, logger):
        self._logger = logger
        self.rv = None
        pass

    # generating BAttM model
    # z' = f(z) + w
    # x = M sigmoid(z) + v
    # w~N(0,q^2I)
    # v~N(0,r^2I)
    # (r estimated by s)
    # input
    #   M : relation between atractor and observation
    #       [col_vec1, col_vec2, ...]
    #   q : dynamic uncertainty
    #   s : sensary uncertainty
    #   blin,blat,g,k,: hopfield parameter
    #   r,o: hopfield parameter
    #   z0,P0: initial
    # out put
    #   model : state space model for BAttM
    # source(paste(pwd, "/R/UKF.R", sep=""))
    # source(paste(pwd, "/R/hopfield.R", sep=""))

    # generating BAttM with general hopfield
    # 一般的な Hopfield で BAttM モデルを作成する
    #  BAttM論文の式(2)に相当?
    #  z' = hopfield_PI(z) + w
    #  BAttM論文の式(3)に相当?
    #  x = obs(z) + v
    # input
    #  att_list : list of attractors
    #  rambda : parameter of hopfield
    #  obs    : observation function
    #  q : dynamic uncertainty
    #  s : sensary uncertainty
    #  注：論文では sensary uncertainty は 'r' で記されているが、
    #      R ソースでは 's' と記されている。
    # BAttM.makeModel_PI = function(att_list,rambda,obs,q,s,z0=NULL,P0=NULL){
    def makeModel_PI(self, att_list, rambda, obs, q, s, z0=None, P0=None):
        #   Dbg.fbegin(match.call()[[1]])
        model = BAttMModel(self._logger, att_list, rambda, obs, q, s, z0, P0)
        model.J, model.tau, model.rambda = self.gen_hopfield_PI(att_list, rambda)
        model.UKF = self.makeUKFModel(model)
        return model


    def makeUKFModel(self, bam_model):
        # 使用パラメータ
        # n, m, Ff, h, Q, R, z0, P0
        obj = UKF.UKF_makeModel(
            bam_model.n,
            bam_model.m,
            bam_model.J,   # オリジナルはラムダ関数 Ff
            bam_model.obs, # オリジナルはラムダ関数 h
            bam_model.Q,
            bam_model.R,
            bam_model.z0,
            bam_model.P0)

        #f = lambda z: ((-1/tau) * z) + (J * tanh(rambda * z))
        return obj


    def gen_hopfield_PI(self, pat_list, rambda, tau=1.0, beta=1.0):
        # X = matrix(unlist(pat_list),ncol=n,byrow = F)
        # 列ごとに一直線に並べる
        X = np.array(pat_list)
        X = X.T
        
        # 行列 X のムーアペンローズ型一般化逆行列を求める
        # Compute the Moore-Penrose pseudo-inverse of a matrix
        # J = beta * X %*% ginv(X)
        J = beta * np.dot(X, np.linalg.pinv(X))
        
        # Dbg.fend(match.call()[[1]])
        # 論文の式(1)に従うが、I は直接外部入力で、今回は未使用。
        # \frac{d}{dt} u が出力になる
        # 論文の u: z
        # 論文の J: J
        # 論文の v: tanh(rambda * z)
        #           式(1) に v = \theta(\lambda u) と定義される
        #           theta は反転可能な写像関数であり、シグモイド関数
        #           等を使用する
        # 論文の I: 未使用
        # function(z){-1/tau * z + J %*% tanh(rambda*z)}
        #f = lambda z: ((-1/tau) * z) + (J * tanh(rambda * z))
        #return f
        # Python で(シリアライズ時に)必要なのは関数 f ではなく、
        # f に渡す行列パラメータ J
        return J, tau, rambda


    # # generating observation function
    # #  obs(z) = sum_i obs_list[[i]] * prod_{j|state_list[[i]][j]=1} sigmoid(z[j]) * prod_{j|state_list[[i]][j]=0} (1-sigmoid(z[j]))
    # # input
    # #  obs_list : list of observations
    # #  state_list : list of states
    # #  r, o : parameters of sigmoid function
    # BAttM.gen_observation = function(obs_list,state_list,r,o=0){
    def gen_observation(self, obs_list, state_list, r, o=0):
        # Dbg.fbegin(match.call()[[1]])
        self._logger.debug("Generating observation function...")
        # n = length(obs_list)
        n = len(obs_list)
        # if(n != length(state_list)){
        #   stop("ERROR : lengths are different between 'obs_list' and 'state_list'")
        # }
        self._logger.debug("obs_list.shape:{} vs state_list.shape:{}".format(obs_list.shape, state_list.shape))
        if n != len(state_list):
            raise Exception("lengths are different between 'obs_list' and 'state_list'")

        # Dbg.fend(match.call()[[1]])
        # 
        #   function(z){
        #     Dbg.fbegin(sprintf("BAttM.gen_observation::obs(%.5f)", z))
        #     ans = 0
        #     # hopfield.R で定義される sigmoid 関数を参照する
        #     res_sig = tanh_sigmoid(z,r,o)
        #     # すべてのアトラクタに対して
        #     for(i in 1:n){
        #       # 大歳先生論文の式(5)
        #       list_1 = state_list[[i]] == 1
        #       list_m1 = state_list[[i]] == -1
        #       id = prod(res_sig[list_1]) * prod(1-res_sig[list_m1])
        #       ans = ans + obs_list[[i]]*id
        #     }
        #     
        #     Dbg.fend("BAttM.gen_observation::obs()")
        #     return(ans)
        #   }
        #   
        # }
        # 関数はシリアライズ化できないので 4 項組タプルで返す
        return obs_list, state_list, r, o


    # # BAttM estimation
    # # input
    # #   model : state space model for BAttM
    # #   x : observed
    # #   estz,Pz : current expectation and covariance
    # # output
    # #   estz',Pz': updated expectation and covariance
    # BAttM.estimation.1step = function(model,x,estz,Pz){
    def estimation_1step(self, cfg, x, estz, Pz):
        #   Dbg.fbegin(match.call()[[1]])
        #   pred = UKF.predict(model,estz,Pz)
        pred = UKF.UKF_predict(cfg, estz, Pz)
        #   UKF.update(model,pred,x)
        result = UKF.UKF_update(cfg, pred, x)
        #   
        #   #Dbg.fend(match.call()[[1]])
        # }
        # 結果はUKFで定義される EstimationResult で estx/estP をメンバーに持つ
        return result


    # ##confidence
    # BAttM.confidence = function(g,z,P){
    def confidence(self, g, z, P):
        #   #Dbg.fbegin(match.call()[[1]])
        #   dmvnorm(g,z,P)
        #   #Dbg.fend(match.call()[[1]])
        # }
        if z is not None and P is not None:
            # cupy 代替関数なし
            result = multivariate_normal.pdf(g, z, P)
        else:
            result = 0.0
            pass

        return result

    # instantiate PDF function
    def mvar_pdf(self, z, P):
        if z is not None and P is not None:
            result = multivariate_normal(z, P)
            self.rv = result
            return self.rv
        else:
            return None


    # PDF function
    def pdf(self, x):
        if self.rv is not None:
            return self.rv.pdf(x)
        else:
            return None

