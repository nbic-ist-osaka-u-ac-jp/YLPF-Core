# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
#
# Copyright (C) 2020 NEC Brain Inspired Computing
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
source(paste(pwd, "/trace_debug.R", sep=""))

#BAttM for time series
#
# * making hopfield function f with given time series
# * remenber the time series with observed value


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
source(paste(pwd, "/UKF.R", sep=""))
source(paste(pwd, "/hopfield.R", sep=""))

library(mvtnorm)
#library(progress)

#BAttM.makeModel = function(M,q,s,blin,blar,g,k,r,o,obs.r,obs.o,z0,P0){
#  Dbg.fbegin(match.call()[[1]])
#  n = ncol(M)
#  m = nrow(M)
#  f = gen.hopfield(n,blin,blat,g,k,r,o)
#  Ff = function(zw){
#    zw[1:n] + f(zw[1:n]) + zw[n + 1:n]
#  }
#  h = function(z){
#    M %*% t(t(sigmoid(z,obs.r,obs.o)))
#  }
#  Q = diag(q^2, n)
#  R = diag(s^2, m)
#
#  UKF.makeModel(n,m,Ff,h,Q,R,z0,P0)
#}

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
BAttM.makeModel_PI = function(att_list,rambda,obs,q,s,z0=NULL,P0=NULL){
  Dbg.fbegin(match.call()[[1]])
  # カテゴリ数 + 1 となるアトラクタ次元数
  n = length(att_list[[1]])  # dimension of attractor
  # 特徴量の次元数
  m = length(obs(att_list[[1]])) # dimension of observation
  # 随時更新されるホップフィールド関数
  # 論文の式(1)及び式(2)に登場する f(z)
  f = gen.hopfield_PI(att_list,1,1,rambda)
  # TODO: UKF論文のUnscented変換で使用される関数?
  Ff = function(zw){
    zw[1:n] + f(zw[1:n]) + zw[n + 1:n]
  }
  # 随時更新される観測関数
  h = function(z){
    obs(z)
  }
  # Q：n×nの対角行列、
  # BAttM.q の二乗で計算される動的な不確実性。
  # 論文 式(2)の直下の Q
  # カテゴリ数=2の場合：
  # [[0.24, 0.00, 0.00],
  #  [0.00, 0.24, 0.00],
  #  [0.00, 0.00, 0.24]]
  Q = diag(q^2, n)
  # R：m×mの対角行列、
  # s の二乗で計算されるセンサー誤差（分散）
  # 論文の p.7 にある $\vec{R} = r^2$ のこと。
  # 特徴量次元数=3の場合：
  # [[0.01, 0.00, 0.00],
  #  [0.00, 0.01, 0.00],
  #  [0.00, 0.00, 0.01]]
  R = diag(s^2, m)
  if(is.null(z0)){
    z0 = rep(0,n)
  }
  if(is.null(P0)){
    # P0：n×nの対角行列
    P0 = diag(1,n)
  }

  Dbg.fend(match.call()[[1]])
  UKF.makeModel(n,m,Ff,h,Q,R,z0,P0)
}



# generating observation function
#  obs(z) = sum_i obs_list[[i]] * prod_{j|state_list[[i]][j]=1} sigmoid(z[j]) * prod_{j|state_list[[i]][j]=0} (1-sigmoid(z[j]))
# input
#  obs_list : list of observations
#  state_list : list of states
#  r, o : parameters of sigmoid function
# TODO: 将来の拡張: 現状、次元数は固定だが、可変にしたい。
BAttM.gen_observation = function(obs_list,state_list,r,o=0){
  Dbg.fbegin(match.call()[[1]])
  
  n = length(obs_list)
  if(n != length(state_list)){
    stop("ERROR : lengths are different between 'obs_list' and 'state_list'")
  }
  
  Dbg.fend(match.call()[[1]])

  function(z){
    Dbg.fbegin(sprintf("BAttM.gen_observation::obs(%.5f)", z))
    ans = 0
    # hopfield.R で定義される sigmoid 関数を参照する
    res_sig = sigmoid(z,r,o)
    # すべてのアトラクタに対して
    for(i in 1:n){
      # 大歳先生論文の式(5)
      list_1 = state_list[[i]] == 1
      list_m1 = state_list[[i]] == -1
      id = prod(res_sig[list_1]) * prod(1-res_sig[list_m1])
      ans = ans + obs_list[[i]]*id
    }
    
    Dbg.fend("BAttM.gen_observation::obs()")
    return(ans)
  }
  
}

# generating observation function
#  obs(z) = sum_i obs_list[[i]] * prod_{j|state_list[[i]][j]=1} sigmoid(z[j]) * prod_{j|state_list[[i]][j]=0} (1-sigmoid(z[j]))
# input
#  obs_list : list of observations
#  state_list : list of states
#  r, o : parameters of sigmoid function
#BAttM.gen_observation2 = function()


# generating attractor
# input
#   traffic pattern list : list of time series whose length s
#   actions
# output
#   M : attractor construction
#
#BAttM.genAtt = function(pat_list){
#  n = length(pat_list)
#  return((matrix(unlist(pat_list),ncol=n)))
#}

# BAttM estimation
# input
#   model : state space model for BAttM
#   x : observed
#   estz,Pz : current expectation and covariance
# output
#   estz',Pz': updated expectation and covariance
BAttM.estimation.1step = function(model,x,estz,Pz){
  Dbg.fbegin(match.call()[[1]])
  
  pred = UKF.predict(model,estz,Pz)
  UKF.update(model,pred,x)
  
  #Dbg.fend(match.call()[[1]])
}

# BAttM estimation reccursive version
# input
#   X : time series of observation(list)
# output
#   list(estz),list(Pz)
#BAttM.estimation = function(model,X){
#  Dbg.fbegin(match.call()[[1]])
#  
#  estz = model$x0
#  Pz = model$P0
#  ans = list()
#
#  pb = progress_bar$new(total = length(X),
#                        format = "[:bar] :percent estimated: :eta (elapsed: :elapsed)",
#                        clear = F)
#  Dbg.info(sprintf("BAttM estimation for %d steps",length(X)))
#  for(i in 1:length(X)){
#    Dbg.info(sprintf("BAttM.estimation.1step for %d step", i))
#    tmp = BAttM.estimation.1step(model,X[[i]],estz,Pz)
#    estz = tmp$estx
#    Pz = tmp$estP
#    ans[[i]]=tmp
#
#    pb$tick()
#  }
#  
#  Dbg.fend(match.call()[[1]])
#  return(ans)
#}

# BAttM prediction
# input
#   est : recursive estimation result
#   start : time for starting prediction
#   h   : horizon length
# output
#   list(z),list(Pz)
#BAttM.prediction = function(model, est, start, h){
#  predz = est[[start]]$estx
#  Pz = est[[start]]$estP
#  ans = list()
#
#  for(i in 1:h){
#    # print(sprintf("%d step",i))
#    pred = UKF.predict(model,predz,Pz)
#    predz = pred$predx
#    Pz = pred$predP
#    ans[[i]] = pred
#  }
#
#  return(ans)
#}

## abstruction
#BAttM.abstruction = function(x){
#  (x - min(x)) / (max(x) - min(x))
#}

##confidence
BAttM.confidence = function(g,z,P){
  #Dbg.fbegin(match.call()[[1]])
  dmvnorm(g,z,P)
  #Dbg.fend(match.call()[[1]])
}
