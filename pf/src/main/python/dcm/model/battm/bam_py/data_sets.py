#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Component: YLPF
# Package  : battm
# Name     : DataSets
# Author   : NBIC
# -------------------------------------------------------------------
#
# Copyright (C) 2018 NEC Brain Inspired Computing
#                    Research Alliance Laboratories
#                    All Rights Reserved.
# -------------------------------------------------------------------

import os
import numpy as np

class DataSets(object):
    def __init__(self, filename, logger):
        """ コンストラクタ """
        self._X = None
        self._N = None
        self.load(filename, logger)
        self._filename = filename
        self._logger = logger

    def load(self, filename, logger):
        """ 入力データセットの読み込み """
        if filename != "" and os.path.exists(filename):
            self.clear()
            X = np.loadtxt(filename, dtype='float', delimiter=',',
                           skiprows=0, ndmin=2)
            if X.size > 0:
                X = X.T
            # データ無しの場合は_N:1次元空配列、_X:2次元空配列となる
            self._N = X[:, 0].astype(int)
            self._X = X[:, 1:]
        return self._N, self._X

    def get(self):
        return self._N, self._X

    def clear(self):
        """ クリア """
        self._X = None
        self._N = None

    def save(self):
        data = np.vstack((np.array([self._N]).astype(np.float), self._X.T))
        np.savetxt(self._filename, data, delimiter=',')

    def delete(self, del_nos):
        idx = np.where(np.array([no in del_nos for no in self._N]))[0]
        self._N = np.delete(self._N, idx)
        self._X = np.delete(self._X, idx, axis=0)

    def update(self, upd_nos, upd_data):
        for upd_idx, no in enumerate(upd_nos):
            data = upd_data[upd_idx]
            idx_ar = np.where(self._N == no)[0]
            if len(idx_ar):
                self._X[idx_ar[0]] = np.array(data).astype(np.float)
            else:
                self._N = np.append(self._N, [no])
                self._X = np.append(self._X, [data], axis=0)
