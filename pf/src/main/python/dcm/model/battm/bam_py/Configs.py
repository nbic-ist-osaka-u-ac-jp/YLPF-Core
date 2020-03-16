#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Component: YLPF
# Package  : battm
# Name     : Configs
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
#    Class: Configs
#     Role: Configuration Variable Class.
# -------------------------------------------------------------------
# 
import logging
import numpy as np
import os
import pickle

class BAttMConfigs(object):
    """ BAttM 用設定クラス """
    def __init__(self):
        #   Bitzer 論文の式(10)の r (シグモイド関数パラメータ)
        self.r = 2.7
        #   Zの更新式のシグモイド関数の傾き
        #   論文中 beta: シグモイド関数パラメータ
        self.rambda = 1.4
        #   ホップフィールドの更新式のノイズ項の分散
        #   論文の q: dynamics uncertainty
        self.q = 0.4898979485566356
        pass


class UKFConfigs(object):
    """ UKF 用 設定クラス """
    def __init__(self):
        #   # UKF パラメータ
        #   # 平均の状態値の周りのシグマ
        #   # 標準偏差ポイントパラメータ
        self.alpha = 0.01
        #   
        #   # UKF パラメータ
        #   # 状態の分布の事前情報
        #   # ガウス分布の場合、beta = 2 が最適。
        self.beta = 2.0
        #   
        #   # UKF パラメータ
        #   # 2 番目のスケーリング パラメーター
        #   # 通常は 0 に設定される。
        #   # 値がさいほど、シグマ ポイントは平均の状態に近くなる。
        #   # 広がりは kappa の平方根に比例する。
        self.kappa = 0.0


class Configs(object):
    """ シリアライズ対象設定情報格納用クラス """
    def __init__(self, logger):
        """ コンストラクタ """
        self._logger = logger
        pass


    # estz/Pz 及び BAttM モデルなど、1 step 毎の動作を実現するために必要な
    # 情報を保持する Configs リスト変数を作成する。
    # Configs.makeConfig = function(cat_pat) {
    def makeConfig(self, cat_pat):
        """ 情報を保持する Configs リスト変数を作成する """
        #   Dbg.fbegin(match.call()[[1]])
        self._logger.debug("Making configuration model.")
        #   ----------------------------------------
        #   バージョン情報
        self.version = 0.1
        #   復元有無
        self.restored = False
        #   デバッグレベル
        self.debug_level = logging.DEBUG
        #   ----------------------------------------
        #   定数群
        #   センサの不確実性 (sensory uncertainty)
        #   SQRT(データ分散) で良いが、事前には分からない
        self.s = 0.10
        # BAM 用初期値
        self.BAttM = BAttMConfigs()
        # UKF 用初期値
        self.UKF = UKFConfigs()
        #   # 論文中 alpha : シグモイド関数パラメータ
        #   config$BAttM.r <- 2.7
        #   
        #   # Zの更新式のシグモイド関数の傾き
        #   # 論文中 beta: シグモイド関数パラメータ
        #   config$BAttM.rambda <- 1.4
        #   
        #   # ホップフィールドの更新式のノイズ項の分散
        #   # 論文の q: dynamics uncertainty
        #   config$BAttM.q <- 0.4898979
        #   
        #   # UKF パラメータ
        #   # 平均の状態値の周りのシグマ
        #   # 標準偏差ポイントパラメータ
        #   config$UKF.alpha <- 0.01
        #   
        #   # UKF パラメータ
        #   # 状態の分布の事前情報
        #   # ガウス分布の場合、beta = 2 が最適。
        #   config$UKF.beta <- 2
        #   
        #   # UKF パラメータ
        #   # 2 番目のスケーリング パラメーター
        #   # 通常は 0 に設定される。
        #   # 値がさいほど、シグマ ポイントは平均の状態に近くなる。
        #   # 広がりは kappa の平方根に比例する。
        #   config$UKF.kappa <- 0
        # 
        #   config <- Configs.updateCategoryPatterns(config, cat_pat)
        self.updatePatterns(cat_pat)
        # 
        #   Dbg.fend(match.call()[[1]])
        self._logger.debug("Configuration model is created.")
        pass


    # # BAttM/UKF のパラメータを更新する. 
    # Configs.updateParams = function(config, params) {
    def updateParams(self, debug_level, s, bamcfg, ukfcfg):
        self._logger.debug("Updating configuration model.")
        #   # デバッグレベル
        #   config$debug_level <- params$debug_level
        self.debug_level = debug_level
        #   Dbg.LEVEL <<- config$debug_level
        self.s            = s
        self.BAttM.r      = bamcfg.r
        self.BAttM.rambda = bamcfg.rambda
        self.BAttM.q      = bamcfg.q
        self.UKF.alpha    = ukfcfg.alpha
        self.UKF.beta     = ukfcfg.beta
        self.UKF.kappa    = ukfcfg.kappa
        pass


    # # カテゴリの平均値情報を更新する.
    def updatePatterns(self, cat_pat):
    # Configs.updateCategoryPatterns = function(config, cat_pat) {
    #   # ----------------------------------------
        self._logger.debug("Updating category patterns.")
        #   # カテゴリパターン数で値が変化する変数群
        #   #
        #   # 特徴量の次元数
        #   # 型：64個の特徴量があるとすれば、64 が入る
        #   config$dim_x <- length(cat_pat[[1]])
        self.dim_x = len(cat_pat[0])
        #   # カテゴリ数
        #   config$num_a <- length(cat_pat)
        self.num_a = len(cat_pat)
        #   # Z 空間の次元数
        #   # 最低でも「カテゴリ数 +1」を指定するのが推奨値
        #   config$dim_z <- config$num_a + 1
        self.dim_z = self.num_a #+ 1

        #   # カテゴリパターン、各カテゴリの代表特徴量値. 
        #   # 型：１行目はカテゴリ番号、２行目以降は特徴量が入る。
        #   # カテゴリの特徴量平均値などが入る。
        #   # 例: データ例(カテゴリ数=3、特徴量の軸数=4の場合)：
        #   #    [[1,    2,    3],
        #   #    [0.00, 0.03, 0.04],
        #   #    [0.00, 0.12, 0.15],
        #   #    [0.03, 0.22, 0.17],
        #   #    [0.25, 0.22, 0.22]]
        #   config$traffic_pat <- cat_pat
        self.traffic_pat = cat_pat
        # 
        #   # ----------------------------------------
        #   # アトラクター(状態)リストの作成
        #   # 型: カテゴリ数 × カテゴリ数の行列
        #   # 例：初期値(カテゴリ数=3の場合)：
        #   # [[+1, -1, -1],
        #   #  [-1, +1, -1],
        #   #  [-1, -1, +1]]
        #   config$state_list = list()
        self.state_list = np.zeros((self.num_a, self.dim_z))
        
        #   for(i in 1:config$num_a){
        for i in np.arange(0, self.num_a):
        #     config$state_list[[i]] <- rep(-1, config$dim_z)
            self.state_list[i,:] = -1
        #     config$state_list[[c(i,i)]] <- 1
            self.state_list[i, i] = 1
        #     Dbg.debug(sprintf("i=%d, dim_z=%d, num_a=%d",
        #             i, config$dim_z, config$num_a))
        #   }
        # 
        self._logger.debug("state_list.shape:{}".format(self.state_list.shape))
        pass


    # # BAttM モデルを更新する. 
    def updateModel(self, BAttM_model):
    # Configs.updateBamModel = function(config, BAttM_model) {
        if self._logger:
            self._logger.debug("Updating BAttM model.")
        #   config$BAttM_model <- BAttM_model
        self.BAttM_model = BAttM_model
        #   config$estz <- BAttM_model$x0
        #   config$Pz <- BAttM_model$P0
        # R では UKF の x0 が参照されているが Python では
        # BAttM モデルの z0 を初期値にする
        self.estz = BAttM_model.z0
        self.Pz = BAttM_model.P0
        pass


# # 現在の BAttM モデルを含めた設定情報をファイル保存(シリアライズ)する.
# Configs.saveConfig = function(cfg) {
def saveConfig(cfg, pkl_name):
    #   Dbg.info(sprintf("Saving configuration file to %s...", CONFIG_FILE_NAME))
    if pkl_name == "":
        return

    # ロガーはシリアライズの対象外、インスタンス生成時に指定する
    # ロガー変数を退避する
    lg = cfg._logger 
    cfg._logger = None
    cfg.BAttM_model._logger = None
    
    with open(pkl_name, "wb") as f:
        pickle.dump(cfg, f)
    
    # 対障外にしたロガーをもとの変数に戻す
    cfg._logger = lg
    cfg.BAttM_model._logger = lg
    pass


# # ファイルに保存されている BAttM モデルを復元(デシリアライズ)する. 
# Configs.restoreConfig = function(cat_pat) {
def restoreConfig(pkl_name, cat_pat, logger):
    if os.path.exists(pkl_name):
    # Dbg.info(sprintf("Restoring configuration file from %s...", CONFIG_FILE_NAME))
        with open(pkl_name, "rb") as f:
            cfg = pickle.load(f)
            cfg.restored = True
            cfg._logger = logger
            cfg.BAttM_model._logger = logger
    else:
        # Dbg.info("Creating new configuration with given category patterns...")
        cfg = Configs(logger)
        cfg.makeConfig(cat_pat)
    return cfg
