#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Component: YLPF
# Package  : battm
# Name     : Normalizer
# Author   : NBIC
# License  : BSD 2-Clause
# -------------------------------------------------------------------
#
# Copyright (C) 2019 NEC Brain Inspired Computing
#                    Research Alliance Laboratories
#                    All Rights Reserved.
# -------------------------------------------------------------------

import numpy as np

from sklearn.preprocessing import MinMaxScaler

from abc import ABCMeta, abstractmethod


class Normalizer(metaclass=ABCMeta):
    def __init__(self):
        self._model = None  # 正規化モデル
        self._raw_ave = None
        self._raw_std = None
        self._cat_labels = None
        self._cat_std = None
        self._cat_ave = None
        self._cat_min = None
        self._cat_max = None

    @abstractmethod
    def normalize_feature_val(self,
                              raw_samples: np.ndarray,
                              labels: list,
                              clip: bool,
                              update: bool):
        """特徴量サンプルの２次元配列を正規化する。update=Trueの場合、正規化モデルを作成・更新後に正規化する。"""

    @abstractmethod
    def normalize_attractor(self, raw_attrs: np.ndarray, clip: bool):
        """アトラクタの代表値の２次元配列を正規化する"""

    @abstractmethod
    def load_stat(self, stat_files: dict):
        """統計量CSVファイルから正規化モデルを作成する"""

    @abstractmethod
    def save_stat(self, stat_files: dict):
        """正規化モデルを再構成するための統計量CSVファイルを作成する"""

    def normalize_sd(self, raw_sd: np.ndarray):
        """ 要素毎の標準偏差(１次元配列)に正規化変換の倍率を適用する"""
        return raw_sd * self._model.scale_

    def _make_stat(self, raw_samples: np.ndarray, ylabel_name: str):
        """ 参考統計値の作成 """
        self._raw_std = np.std(raw_samples, axis=0)
        self._raw_ave = np.average(raw_samples, axis=0)
        # ラベルありの場合、カテゴリ毎の統計値も計算
        if ylabel_name is not None:
            self._make_category_stat(raw_samples, ylabel_name)
        else:
            self._clear_category_stat()

    def _make_category_stat(self, raw_samples: np.ndarray, ylabel_name: str):
        """ カテゴリ別統計値の作成 """
        ylabels = np.loadtxt(ylabel_name, delimiter=",", skiprows=1)
        n_label = np.sort(np.unique(ylabels))
        series = []
        std = []
        ave = []
        min = []
        max = []
        header = []
        v_len = raw_samples.shape[1]
        for l in n_label:
            header.append(str(l))
            indcs = ylabels == l
            std.append(np.std(raw_samples[indcs, :], axis=0))
            ave.append(np.average(raw_samples[indcs, :], axis=0))
            min.append(np.min(raw_samples[indcs, :], axis=0))
            max.append(np.max(raw_samples[indcs, :], axis=0))

        self._cat_labels = header
        self._cat_std = np.array(std)
        self._cat_ave = np.array(ave)
        self._cat_min = np.array(min)
        self._cat_max = np.array(max)

    def _clear_category_stat(self):
        """ カテゴリ別統計値のクリア """
        self._cat_labels = None
        self._cat_std = None
        self._cat_ave = None
        self._cat_min = None
        self._cat_max = None

    def _save_misc_stat(self, stat_files: dict):
        """ 参考統計値のファイル保存 """
        # 全平均、全標準偏差
        if 'ave_std_all' in stat_files:
            ave_std_file = stat_files['ave_std_all']
            ave_std = np.array([self._raw_ave, self._raw_std]).T
            head = "average,std"
            np.savetxt(ave_std_file, ave_std, delimiter=",", header=head)

        # 処理済であれば、カテゴリ別統計値の保存を実施
        if self._cat_labels is not None:
            self._save_category_stat(stat_files)

    def _save_category_stat(self, stat_files: dict):
        """ カテゴリ別統計値のファイル保存 """
        header = ",".join(self._cat_labels)
        # カテゴリ別標準偏差(all列あり)
        if 'stdev_name' in stat_files:
            std_all = np.append(self._cat_std, [self._raw_std], axis=0)
            np.savetxt(stat_files['stdev_name'], std_all.T,
                       delimiter=",", header=header + ',all')
        # カテゴリ別平均
        if 'AVGfile' in stat_files:
            np.savetxt(stat_files['AVGfile'], self._cat_ave.T,
                       delimiter=",", header=header)
        # カテゴリ別最小、最大
        if 'min' in stat_files:
            np.savetxt(stat_files['min'], self._cat_min.T,
                       delimiter=",", header=header)
        if 'max' in stat_files:
            np.savetxt(stat_files['max'], self._cat_max.T,
                       delimiter=",", header=header)

    def _load_misc_stat(self, stat_files: dict):
        """ 参考統計値のファイル読込 """
        # 全平均、全標準偏差（いずれも参考値）
        if 'ave_std_all' in stat_files:
            ave_std_file = stat_files['ave_std_all']
            ave_std = np.loadtxt(ave_std_file, delimiter=",", skiprows=1).T
            self._raw_ave = ave_std[0]
            self._raw_std = ave_std[1]
        else:
            self._raw_ave = None
            self._raw_std = None

        # カテゴリ別があれば読込
        self._clear_category_stat()
        self._load_category_stat(stat_files)

    def _load_category_stat(self, stat_files: dict):
        """ カテゴリ別統計値のファイル読込 """
        # カテゴリ別標準偏差(all列削除)
        if 'stdev_name' in stat_files:
            std = np.loadtxt(stat_files['stdev_name'],
                             delimiter=",", skiprows=1).T
            self._cat_std = std[:-1, :]
        # カテゴリ別平均
        if 'AVGfile' in stat_files:
            self._cat_ave = np.loadtxt(stat_files['AVGfile'],
                                       delimiter=",", skiprows=1).T
        # カテゴリ別最小、最大
        if 'min' in stat_files:
            self._cat_min = np.loadtxt(stat_files['min'],
                                       delimiter=",", skiprows=1).T
        if 'max' in stat_files:
            self._cat_min = np.loadtxt(stat_files['max'],
                                       delimiter=",", skiprows=1).T


class MinMaxNormalizer(Normalizer):
    """ [min, max] -> [0, 1] 変換による正規化を行う """
    def normalize_feature_val(self,
                              raw_samples: np.ndarray,
                              ylabel_name: str,
                              clip: bool,
                              update: bool):
        """
        特徴量サンプルの２次元配列を正規化する。
        update=Trueの場合、正規化モデルを作成・更新後に正規化処理を行う
        """
        if update:
            self._model = MinMaxScaler((0.0, 1.0))
            self._model.fit(raw_samples)
            self._make_stat(raw_samples, ylabel_name)

        return self.normalize_attractor(raw_samples, clip)

    def load_stat(self, stat_files: dict):
        """
        CSVファイルから正規化モデルを作成する
        stat_files['min_max_all']には最小値、最大値ファイルを指定
        stat_files['ave_std_all']には全平均、全標準偏差ファイルを指定
        """
        # 最小値、最大値
        min_max_file = stat_files['min_max_all']
        min_max = np.loadtxt(min_max_file, delimiter=",", skiprows=1).T
        self._model = MinMaxScaler((0, 1))
        self._model.fit(min_max)

        # 参考値
        self._load_misc_stat(stat_files)

    def save_stat(self, stat_files: dict):
        """
        正規化モデルを再構成するためのCSVファイルを作成する
        stat_files['min_max_all']には最小値、最大値ファイルを指定
        stat_files['ave_std_all']には全平均、全標準偏差ファイルを指定
        """
        # 最小値、最大値
        min_max_file = stat_files['min_max_all']
        min_max = np.array([self._model.data_min_, self._model.data_max_]).T
        head = "min,max"
        np.savetxt(min_max_file, min_max,
                   delimiter=",", header=head)

        # 参考値
        self._save_misc_stat(stat_files)

    def normalize_attractor(self, raw_attrs: np.ndarray, clip: bool):
        """
        アトラクタの代表値を正規化する
        """
        # データ無しの場合は、同じshapeの空配列を返却する。
        if raw_attrs.size == 0:
            return raw_attrs.copy()
        norm_data = self._model.transform(raw_attrs)
        if clip:
            min, max = self._model.get_params()['feature_range']
            norm_data = norm_data.clip(min, max)
        return norm_data


class ZeroMaxNormalizer(Normalizer):
    """ [0, max] -> [0, 1] 変換による正規化を行う """
    def __init__(self):
        super().__init__()
        self._raw_min = None

    def normalize_feature_val(self,
                              raw_samples: np.ndarray,
                              ylabel_name: str,
                              clip: bool,
                              update: bool):
        """
        特徴量サンプルの２次元配列を正規化する。
        update=Trueの場合、正規化モデルを作成・更新後に正規化処理を行う
        """
        if update:
            self._model = MinMaxScaler((0, 1))
            # ゼロデータを追加してモデル作成
            zero_max = [np.zeros(raw_samples.shape[1]),
                        np.max(raw_samples, axis=0)]
            self._model.fit(zero_max)
            self._raw_min = np.min(raw_samples, axis=0)
            self._make_stat(raw_samples, ylabel_name)

        return self.normalize_attractor(raw_samples, clip)

    def load_stat(self, stat_files: dict):
        """
        CSVファイルから正規化モデルを作成する
        stat_files['min_max_all']には最小値、最大値ファイルを指定
        stat_files['ave_std_all']には全平均、全標準偏差ファイルを指定
        """
        # 最小値、最大値
        min_max_file = stat_files['min_max_all']
        min_max = np.loadtxt(min_max_file, delimiter=",", skiprows=1).T
        self._model = MinMaxScaler((0, 1))
        self._model.fit([np.zeros(min_max.shape[1]), min_max[1]])
        self._raw_min = min_max[0]

        # 参考値
        self._load_misc_stat(stat_files)

    def save_stat(self, stat_files: dict):
        """
        正規化モデルを再構成するためのCSVファイルを作成する
        stat_files['min_max_all']には最小値、最大値ファイルを指定
        stat_files['ave_std_all']には全平均、全標準偏差ファイルを指定
        """
        # 最小値、最大値
        min_max_file = stat_files['min_max_all']
        min_max = np.array([self._raw_min, self._model.data_max_]).T
        head = "min,max"
        np.savetxt(min_max_file, min_max,
                   delimiter=",", header=head)

        # 参考値
        self._save_misc_stat(stat_files)

    def normalize_attractor(self, raw_attrs: np.ndarray, clip: bool):
        """
        アトラクタの代表値を正規化する
        """
        norm_data = self._model.transform(raw_attrs)
        if clip:
            min, max = self._model.get_params()['feature_range']
            norm_data = norm_data.clip(min, max)
        return norm_data
