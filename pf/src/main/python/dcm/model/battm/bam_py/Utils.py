#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Component: YLPF
# Package  : battm
# Name     : Utils
# Author   : NBIC
# License  : BSD 2-Clause
# -------------------------------------------------------------------
#
# Copyright (C) 2019 NEC Brain Inspired Computing
#                    Research Alliance Laboratories
#                    All Rights Reserved.
# -------------------------------------------------------------------
#
import os
import numpy as np

# 結果(確信度)をファイルに出力する.
def WriteResult(x, fname):
    # CSV用データを整形する
    # 追記モードで開く
    if os.path.exists(fname):
        result = np.loadtxt(
            fname,
            dtype='float',
            delimiter=",")
        # 末尾に追加
        result = np.vstack((result, x))
    else:
        result = x

    np.savetxt(
        fname,
        result,
        delimiter=",")

    pass

