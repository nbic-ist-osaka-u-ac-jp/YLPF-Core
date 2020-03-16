#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Component: YLPF
# Package  : battm
# Name     : YLabels
# Author   : NBIC
# License  : BSD 2-Clause
# -------------------------------------------------------------------
#
# Copyright (C) 2018 NEC Brain Inspired Computing
#                    Research Alliance Laboratories
#                    All Rights Reserved.
# -------------------------------------------------------------------

import os
import numpy as np

class YLabels(object):
    """ 正解ラベル表現クラス """
    def __init__(self, filename, logger):
        """ コンストラクタ """
        self._ymap = {}
        self._ymap_raw = {}
        self._filename = filename
        self.load(filename, logger)

    def load(self, filename, logger):
        """ ラベルファイルを読み込む """
        if filename != "" and os.path.exists(filename):
            self.clear()
            logger.debug("Loading y-label file: {} ...".format(filename))
            Y = np.loadtxt(filename, delimiter=",", skiprows=0, ndmin=2)\
                .astype(np.int).T
            # 通常のラベル情報は、アトラクタ候補情報を含めない -1～98の値とする
            # アトラクタ候補情報100の倍数を加算することする。
            # -1（未分類）はアトラクタ候補情報を含むことがある。
            # -99以下（無効データ）はアトラクタ候補情報を含めることができない前程
            for k, v in Y:
                self._ymap[k] = (v + 1) % 100 - 1 if v >= -1 else v
                self._ymap_raw[k] = v
        else:
            logger.debug("There is no valid ylabel file:{}".format(filename))

        return self._ymap

    def save(self, upd_map={}):
        """ ラベルファイルを書き込む """
        ymap_new = self._ymap_raw.copy()
        ymap_new.update(upd_map)
        data = [list(ymap_new.keys()), list(ymap_new.values())]
        np.savetxt(self._filename, data, delimiter=",")
        # リロード相当処理
        self._ymap_raw = ymap_new
        for k in upd_map:
            v = upd_map[k]
            self._ymap[k] = (v + 1) % 100 - 1 if v >= -1 else v

    def delete_proposal(self):
        """ アトラクタ候補情報を削除する """
        self._ymap_raw = self._ymap.copy()

    def clear(self):
        """ ラベル情報をクリアする """
        self._ymap.clear()
        self._ymap_raw.clear()

    def getKeys(self, raw=False):
        """ キー値を取得する """
        ymap = self._ymap_raw if raw else self._ymap
        if len(ymap) == 0:
            return ['']
        else:
            return np.array(list(ymap.keys()))

    def getValues(self, uniq=False, raw=False):
        """ ラベル値を取得する """
        ymap = self._ymap_raw if raw else self._ymap
        if len(ymap) == 0:
            return ['']
        elif uniq:
            return np.unique(list(ymap.values()))
        else:
            return np.array(list(ymap.values()))

    def getItems(self, raw=False):
        """ dict を返す """
        ymap = self._ymap_raw if raw else self._ymap
        return ymap

    def getIndices(self, label, N, raw=False):
        """ 指定されたラベルの配列Indexを取得する """
        ymap = self._ymap_raw if raw else self._ymap
        if label == '':
            return [True for x in N]
        
        ind_list = []
        for k, v in ymap.items():
            if v == label:
                ind_list.append(k)
        indcs = [x in ind_list for x in N]
        return indcs

    def getIndRange(self, labels, N, raw=False):
        """ 指定されたラベル範囲の配列Indexを取得する """
        ymap = self._ymap_raw if raw else self._ymap
        if labels is None:
            return [True for x in N]

        ind_list = []
        min_label = min(labels)
        max_label = max(labels)
        for k, v in ymap.items():
            if min_label <= v <= max_label:
                ind_list.append(k)
        indcs = [x in ind_list for x in N]
        return indcs
