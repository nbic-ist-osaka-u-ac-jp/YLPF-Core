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
#    Class: UKF
#     Role: UKF Class.
# -------------------------------------------------------------------
# 
# 
source(paste(pwd, "/trace_debug.R", sep=""))

# unscented kalman filter(UKF)
# UKF の論文: https://doi.org/10.1109/ASSPCC.2000.882463

#DEBUG = F

# system model
# x' = f(x,w)  ; n-vector
# y = h(x) + v ; m-vector
#
# given:
#  E[w]=0, V[w]=Q
#  E[v]=0, V[v]=R
#  initial state: x0|0, P0|0

# UKF のモデルを作成する
# BAttM モデル本体、Save/Restore 対象
UKF.makeModel = function(n,m,f,h,Q,R,x0,P0){
  Dbg.fbegin(match.call()[[1]])
  
  model = list()
  model$n = n
  model$m = m
  model$f = f
  model$h = h
  model$Q = Q
  model$R = R
  model$x0 = x0
  model$P0 = P0
  
  Dbg.fend(match.call()[[1]])
  return(model)
}

# xe : x_k-1|k-1
# Pe : P_k-1|k-1
UKF.predict = function(model,xe,Pe){
  Dbg.fbegin(match.call()[[1]])
  
  num.w = sqrt(length(model$Q))
  #  estx：1×nの行列。
  # 意思決定状態 z の期待値 (est[[t]]$estx)
  # 各カテゴリのzの期待値が格納されている。
  estX = c(xe,rep(0,num.w))

  # estP：n×xの共分散行列。
  # 共分散行列：est[[t]]$estP
  # カテゴリ間の共分散が格納されている。
  estP = cbind(Pe,matrix(0,ncol=num.w,nrow=model$n))
  estP = rbind(estP,cbind(matrix(0,ncol=model$n,nrow=num.w),model$Q))

  #sampling
  # sigma点を計算する
  samples.X = UKF.sample(estX,estP,model$n+num.w)
  # 
  weights.s = UKF.weights.s(model$n + num.w)
  weights.c = UKF.weights.c(model$n + num.w)

  #U-transformation
  samples.x2 = UKF.Utransformation(samples.X,model$f)
  predx = list.weightedSum(samples.x2,weights.s)
  predP = list.weightedSquareSum(lapply(samples.x2, function(x){return(x-predx)}),weights.c)

  Dbg.fend(match.call()[[1]])
  return(list(predx=predx,predP=predP,samples=samples.x2))
}

UKF.update = function(model,pred,y){
  Dbg.fbegin(match.call()[[1]])

  num.v = sqrt(length(model$R))
  num.w = sqrt(length(model$Q))
  num.x = length(pred$predx)

  # predX = c(pred$predx,rep(0,num.v))
  # predP = cbind(pred$predP,matrix(0,ncol=num.v,nrow=num.x))
  # predP = rbind(predP,cbind(matrix(0,ncol=num.x,nrow=num.v),model$R))

  #sampling
  # samples.X = UKF.sample(predX,predP,model$n)
  # weights.s = UKF.weights.s(model$n)
  # weights.c = UKF.weights.c(model$n)
  samples.X = pred$samples
  weights.s = UKF.weights.s(model$n + num.w)
  weights.c = UKF.weights.c(model$n + num.w)

  # U-transformation 
  # Sigma点列の広がり具合を決めるために Unscented 変換する
  samples.y = UKF.Utransformation(samples.X,model$h)
  predy = list.weightedSum(samples.y,weights.s)
  predPyy = list.weightedSquareSum(lapply(samples.y,function(x){return(x-predy)}),weights.c) + model$R
  predPxy = list.weightedCrossSum(lapply(pred$samples,function(x){return(x-pred$predx)}),
                                lapply(samples.y,function(x){return(x-predy)}),weights.c)

  #Kalman gain
  K = predPxy %*% solve(predPyy)

  #update
  estx = pred$predx + K %*% (y-predy)
  estP = pred$predP - K %*% predPyy %*% t(K)

  #if(DEBUG){
  #  print("y-predy: ")
  #  print(y-predy)
  #  print("kalman gain: ")
  #  print(K)
  #}

  Dbg.fend(match.call()[[1]])
  return(list(estx=estx,estP=estP))
}

#UKF.estimation.recursive = function(model,y,x0,P0){
#  Dbg.fbegin(match.call()[[1]])
#
#  x = x0
#  P = P0
#  ans = list()
#  for(i in 1:length(y)){
#    pred = UKF.predict(model,x,P)
#    est = UKF.update(model,pred,y[[i]])
#    x = est$estx
#    P = est$estP
#    ans[[i]] = list(x = x,P = P)
#  }
#  
#  Dbg.fend(match.call()[[1]])
#  return(ans)
#}

# Sigma点列を計算する
UKF.sample = function(estX,estP,n){
  Dbg.fbegin(match.call()[[1]])
  
  x = list()
  idx.x = 1

  estSigma = matrix.sqrt(estP)
  # UKFの論文 p.3 左中央の式
  # \lambda = alpha^2 (L + \kappa) - Lに従う
  lambda = Cfg$UKF.alpha^2 * (n + Cfg$UKF.kappa) - n

  # UKFの論文 p.3 左中央の式(15)の上から三つに従う
  x[[idx.x]] = estX
  idx.x = idx.x + 1
  for(i in 1:n){
    x[[idx.x]] = estX + sqrt(n+lambda)*estSigma[i,] # + sigma
    idx.x = idx.x + 1
    x[[idx.x]] = estX - sqrt(n+lambda)*estSigma[i,] # - sigma
    idx.x = idx.x + 1
  }
  
  Dbg.fend(match.call()[[1]])
  return(x)
}

# 観測関数で Unscented 変換を行う
UKF.Utransformation = function(samples,f){
  Dbg.fbegin(match.call()[[1]])

  # ans = list()
  # for(i in 1:length(samples)){
  #   ans[[i]] = f(samples[[i]])
  # }
  # return(ans)
  
  Dbg.fend(match.call()[[1]])
  # リストの各要素に対して観測関数を適用する
  lapply(samples,f)
}

# Sigma点列用のウェイト W^{(m)}_0 を計算する
UKF.weights.s = function(n){
  Dbg.fbegin(match.call()[[1]])

  lambda = Cfg$UKF.alpha^2 * (n + Cfg$UKF.kappa) - n
  
  # UKFの論文 p.3 左中央の式(15)の下三つに従う
  # Sigma点列に対するウェイトを計算する
  # W^{(m)}_0 = \lambda / (L + \lambda)
  ws = lambda / (n + lambda)
  
  for(i in 1:n){
    ws = c(ws,1/(2*(n+lambda))) # weights for +sigma
    ws = c(ws,1/(2*(n+lambda))) #             -sigma
  }
  
  Dbg.fend(match.call()[[1]])
  return(ws)
}

# Sigma点列用のウェイト W^{(c)}_0 えを計算する
UKF.weights.c = function(n){
  Dbg.fbegin(match.call()[[1]])
  
  lambda = Cfg$UKF.alpha^2 * (n + Cfg$UKF.kappa) - n
  
  # W^{(c)}_0 = \lambda / (L + \lambda) + (1 - \alpha^2 + \beta)
  ws = lambda / (n + lambda) + (1-Cfg$UKF.alpha^2+Cfg$UKF.beta)
  
  for(i in 1:n){
    ws = c(ws,1/(2*(n+lambda))) # weights for +sigma
    ws = c(ws,1/(2*(n+lambda))) #             -sigma
  }
  
  Dbg.fend(match.call()[[1]])
  return(ws)
}

# 行列の平方根を求める
matrix.sqrt = function(A){
  Dbg.fbegin(match.call()[[1]])
  
  # 特異値分解(SVD)を行う
  # X = U %*% D %*% V を満たす U, D, V を求める
  tmp = svd(A)
  
  # 直行行列 U
  U = tmp$u
  
  # 直行行列 V
  V = tmp$v
  
  # X の特異値を対角成分とする対角行列 D
  # 単位行列を作成する
  D = diag(sqrt(tmp$d))
  
  Dbg.fend(match.call()[[1]])
  return( U %*% D %*% t(V))
}

# リストの重み付き和を計算する
# ans = \sum^{n}_{i=1} l_i \times w_i
list.weightedSum = function(l,weights){
  Dbg.fbegin(match.call()[[1]])
  
  ans = 0
  for(i in 1:length(l)){
    ans = ans + l[[i]] * weights[i]
  }
  
  Dbg.fend(match.call()[[1]])
  return(ans)
}

# リストの重み付き平方和をもとめる
# ans = \sum^{n}_{i=1} l_i * {}^t l_i \times w_i
list.weightedSquareSum = function(l,weights){
  Dbg.fbegin(match.call()[[1]])
  
  ans = 0
  for(i in 1:length(l)){
    ans = ans + weights[i] * l[[i]] %*% t(l[[i]])
  }
  
  Dbg.fend(match.call()[[1]])
  return(ans)
}

# リストの重み付き直交和をもとめる
# ans = \sum^n_{i=1} (l1_i * {}^t l2_i ) \times w_i
list.weightedCrossSum = function(l1,l2,weights){
  Dbg.fbegin(match.call()[[1]])
  
  ans = 0
  for(i in 1:length(l1)){
    ans = ans + weights[i] * l1[[i]] %*% t(l2[[i]])
  }
  
  Dbg.fend(match.call()[[1]])
  return(ans)
}
