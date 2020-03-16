#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Component: YLPF
# Package  : battm
# Name     : sigmoid
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
#    Class: Hopfield
#     Role: Hopfield Class.
# -------------------------------------------------------------------
#
#
import math
import numpy as np
import scipy.special

# シグモイド関数
# 状態保存の必要なし
# 論文では exp を使っているが、シグモイド関数 \varsigma_{a}(x)
# は {\frac{1}{1+e^{-ax}}} = {\frac {\tanh(ax/2)+1}{2}}
# として、tanh 式と等価である
# sigmoid = function(z,r,o){
def tanh_sigmoid(z, r, o):
    x = r * (z - o)
    res = scipy.special.expit(x)
    return res

def cp_tanh_sigmoid(z, r, o):
    res = (cp.tanh(r * (z - o) * 0.5) + 1.0) * 0.5
    return res

