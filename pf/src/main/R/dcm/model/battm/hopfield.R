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
#    Class: Hopfield
#     Role: Hopfield Class.
# -------------------------------------------------------------------
# 
# 
source(paste(pwd, "/trace_debug.R", sep=""))

library(MASS)

# generating a hop field function
# x' = f(x) = k(L * sigmoid(z) + blin(g*mat(1) - z))
# L = blat(I-mat(1))

# input:
#   n : number of attractor
#   blin:
#   g: goal state
#   blat:
#   k:
#   r,o: sigmoid parameter
#
#gen.hopfield = function(n,blin,blat,g,k,r,o){
#  L = blat * (diag(1,n) - matrix(1,ncol = n,nrow = n))
#  function(z){k*(L %*% sigmoid(z,r,o) + blin * (g * rep(1,n) - z))}
#}

# generating a hop field function (pseudoinverse type)
# z' = -1/tau z + beta XX^+ sigmoid(z) (no input)
# (http://www.math.colostate.edu/~shipman/47/volume12009/zhang.pdf)
# input:
#  pat_list : list of patterns to remember
#  tau,beta : rate
#  rambda : sigmoid parameter
#
# 状態保存の必要なし
gen.hopfield_PI = function(pat_list,tau=1,beta=1,rambda){
  Dbg.fbegin(match.call()[[1]])
  n = length(pat_list)
  X = matrix(unlist(pat_list),ncol=n,byrow = F)
  J = beta * X %*% ginv(X)
  Dbg.fend(match.call()[[1]])
  # 論文の式(1)に従うが、I は直接外部入力で、今回は未使用。
  # \frac{d}{dt} u が出力になる
  # 論文の u: z
  # 論文の J: J
  # 論文の v: tanh(rambda * z)
  #           式(1) に v = \theta(\lambda u) と定義される
  #           theta は反転可能な写像関数であり、シグモイド関数
  #           等を使用する
  # 論文の I: 未使用
  function(z){-1/tau * z + J %*% tanh(rambda*z)}
}

# generating a limit cycle hopfield
#
# input:
# att_list : list of attractor
# nxt_list : list of predecessor of each att_list[i]
# tau,beta  :rate parameter
# rambda : sigmoid parameter
#gen.hopfield_LC = function(att_list,nxt_list,tau=1,beta=1,rambda){
#  Dbg.fbegin(match.call()[[1]])
#  n = length(att_list)
#  m = length(nxt_list)
#  if(n != m){
#    stop(sprintf("different lengths between att_list:%d and nxt_list:%d",n,m))
#  }
#  X = matrix(unlist(att_list),ncol=n,byrow=F)
#  F = matrix(unlist(nxt_list),ncol=n,byrow=F)
#  J = beta * F %*% ginv(X)
#  function(z){-1/tau*z + J %*% tanh(rambda*z)}
#}

## https://arxiv.org/pdf/1308.5201.pdf
##input:
## att_list
## nxt_list
## beta
## C0
## lambda
## tau
#gen.hopfield_LC = function(att_list,nxt_list,beta,C0,lambda){
#  n = length(att_list)
#  m = length(nxt_list)
#  if(n != m){
#    stop(sprintf("different lengths between att_list:%d and nxt_list:%d",n,m))
#  }
#
#  beta_k = atanh(beta)/ (lambda*beta)
#  C1 = 1-C0
#
#  X = matrix(unlist(att_list),ncol=n,byrow=F)
#  F = matrix(unlist(nxt_list),ncol=n,byrow=F)
#
#  J0 = X %*% ginv(X)
#  J = F %*% ginv(X)
#
#  function(z){-z + C0*beta_k* J0 %*% tanh(lambda * z) + C1 * beta_k * J %*% tanh(lambda * z)}
#}

# sigmoid function
# sigmoid(z) = {1+exp(-r(zi-o))}^-1
#sigmoid = function(z,r,o){
#  tmp.fun = function(zi){
#    1/(1+exp(-r*(zi-o)))
#  }
#
#  apply(matrix(z),1,tmp.fun)
#}

# シグモイド関数
# 状態保存の必要なし
# 論文では exp を使っているが、シグモイド関数 \varsigma_{a}(x)
# は {\frac{1}{1+e^{-ax}}} = {\frac {\tanh(ax/2)+1}{2}}
# として、tanh 式と等価である
sigmoid = function(z,r,o){
  # 1/(1+exp(-r*(z-o)))
  (tanh(r*(z-o)*0.5)+1) * 0.5
}

