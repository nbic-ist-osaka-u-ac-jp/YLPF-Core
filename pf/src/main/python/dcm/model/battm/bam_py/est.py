#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Component: YLPF
# Package  : battm
# Name     : Estimator
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
#    Class: Estimator
#     Role: Estimator Class.
# -------------------------------------------------------------------
# 
# バージョン情報.
Python_BAttM_ver = "2020.02.21.1500"

import csv
import json
import math
import numpy as np
import os
import re
import shutil
import copy
import datetime
import sys
import pickle
import logging

from trace_debug import myLogger
from Configs import Configs, saveConfig, restoreConfig, BAttMConfigs, UKFConfigs
from BAttM4ts import BAttM
from Utils import WriteResult
from norm import MinMaxNormalizer
from args_helper import ArgsHelper
from args_members import ArgsMembers
from data_sets import DataSets
from ylabels import YLabels

class BattmEstimator(object):
    def __init__(self):
        self._cbam = None
        self._zbam = None
        self._X = None
        self._AVG = None
        pass

    def init_data(self, args):
        # 観測刺激リストファイル名を取得する.
        # Python から呼び出されている場合は直接変数 X をアサイン
        # するため、ファイルは使用しない.
        # 型：１行目は「カテゴリ番号」、2行目以降は特徴量。
        # 観測データ X_i が増えた場合、列を追加する。
        # 例：データ例（観測回数=4、特徴量の軸数=4の場合)：
        # [[1,    2,    2,    10],
        # [0.00, 0.08, 0.00, 0.26],
        # [0.03, 0.20, 0.01, 0.28],
        # [0.08, 0.30, 0.16, 0.18],
        # [0.28, 0.26, 0.16, 0.18]]
        if args.cli_mode > 0:
             INPUT_DATA = args.Xfile

        # カテゴリの平均値ファイル名を取得する.
        if args.cli_mode > 0:
            PATTERN_DATA = args.AVGfile

        return INPUT_DATA, PATTERN_DATA

    def init_configs(self, args, logger, cat_patterns):
        # 以前の設定が保存されている場合はリストアする.
        # 存在しない場合は新規に作成する.
        Cfg = Configs(logger)
        Cfg.makeConfig(cat_patterns)

        # BAttM/UKF のパラメータを更新する. 
        bam_cfg = BAttMConfigs()
        bam_cfg.r = args.bam_r
        bam_cfg.rambda = args.bam_rambda
        bam_cfg.q = args.bam_q
 
        ukf_cfg = UKFConfigs()
        ukf_cfg.alpha = args.ukf_alpha
        ukf_cfg.beta = args.ukf_beta
        ukf_cfg.kappa = args.ukf_kappa

        Cfg.updateParams(args.level, args.bam_s, bam_cfg, ukf_cfg)

        return Cfg

    def init_cat_patterns(self, args, logger):
        # カテゴリの平均値データを Python 変数から取得する.
        if args.normalize > 0:
            cat_patterns = self._AVG
        elif args.cli_mode > 0:
            cat_patterns = np.loadtxt(args.AVGfile, dtype='float', delimiter=',', skiprows=1)
            # Python では転置行列を使用する
            cat_patterns = cat_patterns.T
            if len(cat_patterns.shape) == 1:
                cat_patterns = np.array([cat_patterns]).T
            logger.info("Loaded {} shape category patterns from {}.".format(cat_patterns.shape, args.AVGfile))

        # 特徴量選択
        logger.debug("Category pattern shape:{}".format(cat_patterns.shape))
        indices = args.select_feat
        if ',' in indices:
            indices = '(' + indices + ')'
        cat_patterns = eval("cat_patterns[:," + indices + "]")
        logger.debug("Category pattern shape:{}".format(cat_patterns.shape))
        return cat_patterns

    def get_stimulus(self, args, logger, filename):
        X = None
        if args.cli_mode > 0:
            if args.normalize > 0:
                X = self._X_valid if args.train_only else self._X_test
            elif os.path.exists(filename):
                logger.info("Reading stimulus from %s..." % filename)
                X = np.loadtxt(filename, dtype='float', delimiter=',', skiprows=1)
                # Python では転置行列を使用する
                X = np.array([X]) if X.ndim == 1 else X.T
                logger.info("Loaded %d items in stimulus file." % len(X))

        # 入力刺激の選択
        logger.debug("Input data shape:{}".format(X.shape))
        X = eval("X[" + args.select_data + "]")

        XN, _ = DataSets(args.valid_file if args.train_only else args.test_file, logger).get()
        XN = eval("XN[" + args.select_data + "]")

        # ラベル情報を抽出
        label_name = args.vlabel_name if args.train_only else args.ylabel_name
        ylabels = YLabels(label_name, logger)
        label_keys = ylabels.getKeys()
        label_vals = ylabels.getValues()
        L_enable_nos = label_keys[label_vals >= -1]

        # Xのデータ番号を元に、抽出位置を特定
        X_inds = [XN_ in L_enable_nos for XN_ in XN]
        X = X[X_inds]

        logger.debug("Input data shape:{}".format(X.shape))

        # 特徴量の選択
        indices = args.select_feat
        if ',' in indices:
            indices = '(' + indices + ')'
        X = eval("X[:," + indices + "]")
        logger.debug("Input data shape:{}".format(X.shape))

        return X

    def test1step(self, args, logger, bam, Cfg, i, X, results):
        BAttM_res = results
        # 初回以外で Python 呼び出しではない場合、設定情報をリストアする.
        if i > 0 and args.save_cfg > 0:
            logger.debug("Restoring configuration by using {}.".format(args.pkl_name))
            Cfg = restoreConfig(args.pkl_name, cat_patterns, logger)

        # 基本的には BAttM モデルの 1step に入力する刺激X_iをオンライン入力にすれば良い.
        logger.info("BAttM.estimation.1step for {} step".format(i))

        # 1 ステップ毎に推定する. 
        # Z_{t-1}、観測刺激 X_t、現在の BAttM モデルを入力として
        # 時刻 t における推定 x_t 及び推定 P_t を求める
        next_est = bam.estimation_1step(Cfg, X[i], Cfg.estz, Cfg.Pz)

        # t-1 の estz/Pz 変数を t 時点の値で上書きする
        # estx: 意思決定状態 z の期待値
        Cfg.estz = next_est.estx
        # estP: 意思決定状態 z の共分散行列
        Cfg.Pz = next_est.estP

        # 結果をリストに格納する
        if args.cli_mode > 0:
            BAttM_res.append(next_est)

        # 結果を正規化する.
        prob_stat = np.zeros(len(Cfg.state_list))

        # 全カテゴリに対して確率を計算(正規化)する
        mvpdf = bam.mvar_pdf(next_est.estx, next_est.estP)
        for c in np.arange(0, len(Cfg.state_list)):
            tmp = Cfg.state_list[c]
            # dmvnorm(x, mu, Sigma) 関数は多変量正規分布の確率密度関数を計算する
            # 引数:
            #      x: 多変量観測値の vector/matrix
            #     mu: 平均値の vector/matrix
            #  sigma: 平方共分散行列
            # 
            # ここでは x に tmp を指定している。
            # tmp は本来得たい、各カテゴリであると意思決定する確率 1.0 を
            # 多変量観測値のベクトルにする
            #  [1]  1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
            # 
            # mu は 推定される観測値 X' (estx) を観測値の平均値とみなしている
            #  [1,]  0.7508208
            #  [2,] -1.1424043
            #    :     : 
            # [10,] -1.1280827
            # [11,] -1.1888434
            # 
            # estP: アトラクタ共分散を sigma として指定する
            #                [,1]          [,2] ...   [,11]
            #  [1,]  2.558807e-01  0.0004622708 ...  0.003042178
            #  [2,]  4.622708e-04  0.2402536165 ... -0.009935292
            #   :         :             :                : 
            # [11,]  3.042178e-03 -0.0099352925 ...  0.229027727
            # 
            # 上記の入力 x=tmp, mu=estx, Sigma=estP を指定して、
            # 各カテゴリの確率 prob_stat[i] を計算する(正規化)
            # [1] 0.04839855
            prob_stat[c] = bam.pdf(tmp)

        if args.result_file != "" and args.no_out == 0:
            WriteResult(prob_stat, args.result_file)
            saveConfig(Cfg, args.pkl_name)
        pass

    def save_cfg(self, args, logger):
        """ 設定ファイル出力(未使用) """
        if args.json_file == "":
            return

        cfg_dict = {}
        if os.path.exists(args.json_file):
            with open(args.json_file, mode='r') as f:
                json_str = f.read()
                if len(json_str):
                    cfg_dict = json.loads(json_str)

        with open(args.json_file, mode="w") as f:
            cfg_dict.update(args.__dict__)
            jsn_obj = json.dumps(cfg_dict, indent=2, separators=(',', ': '))
            f.write(jsn_obj)
            f.write("\n")

    def save_results(self, args, logger, bam, Cfg, X, BAttM_res):
        # 状態を保存
        saveConfig(Cfg, args.pkl_name)

        logger.info("Formatting results...")

        confBAttM = list()

        # 全カテゴリの確信度を格納する
        for i in np.arange(0, len(Cfg.state_list)):
            rows = list()
            tmp = Cfg.state_list[i]
            # estx/estP をもとにして確信度を計算する
            # （各step毎）に関数を適用して、結果をリストに格納する
            cnt = 0
            for res in BAttM_res:
                conf_res = bam.confidence(tmp, res.estx, res.estP)
                # リストを一つのベクトルにまとめ、行列に連結する
                rows.append(conf_res)
                cnt += 1
            confBAttM.append(rows)

        confBAttM = np.array(confBAttM).T

        zBAttM = list()

        pBAttM = list()
        pBAttM.append(np.arange(1, 1 + len(Cfg.state_list)))

        for i in np.arange(0, len(X) * args.repeat_cnt):
            #   # 観測刺激の予測値 z の結果を行列に格納する
            if BAttM_res[i].estx is not None:
                zBAttM.append(BAttM_res[i].estx)
            #   # 各列の合計値を求め、アトラクタ毎の共分散の和を計算する
            p = BAttM_res[i].estP
            if p is not None:
                tmp = BAttM_res[i].estP * BAttM_res[i].estP
                res = np.sum(tmp, axis=1)
                pBAttM.append(res)

        # 結果を出力
        logger.info("Writing CSVs...")
        # 確信度: 各時刻、各カテゴリの確信度を格納した行列 
        # 値範囲は0-1 ではない
        cidx = np.array([np.arange(1, 1 + len(Cfg.state_list))], dtype=float)
        cdata = np.vstack((cidx, np.array(confBAttM)))
        np.savetxt(args.conf_file, cdata, delimiter=",")
        # z空間の位置 (観測刺激の予測値に等しい)
        zidx = np.array([np.arange(1, 1 + len(zBAttM))])
        zdata = np.hstack((zidx.T, np.array(zBAttM)))
        np.savetxt(args.z_file, zdata.T, delimiter=",")
        # アトラクタ毎の共分散和
        np.savetxt(args.p_file, np.array(pBAttM), delimiter=",")
        # 最大確信度のラベルIndex( 1 はじまり)
        ylabels_conf = confBAttM.argmax(axis=1) + 1
        maxes = confBAttM.max(axis=1)
        ylabels_conf[maxes==0.0] = -1
        if args.repeat_cnt > 1:
            idcs = np.arange(args.repeat_cnt - 1, len(ylabels_conf), args.repeat_cnt)
        else:
            idcs = np.arange(0, len(ylabels_conf))

        filename = args.valid_file if args.train_only else args.test_file
        N, _ = DataSets(filename, logger).get()
        feat_nos = eval('N[' + args.select_data + ']')

        # 特徴量のデータ番号とラベル定義を参照し
        # BAM推論されたデータ番号を推定してiBAttM.csvのヘッダに設定する。
        label_name = args.vlabel_name if args.train_only else args.ylabel_name
        ylabels = YLabels(label_name, logger)
        label_keys = ylabels.getKeys()
        label_keys = eval("label_keys[" + args.select_data + "]")
        label_vals = ylabels.getValues()
        label_vals = eval("label_vals[" + args.select_data + "]")
        L_enable_nos = label_keys[label_vals >= -1]

        # Xのデータ番号を元に、抽出データを特定
        X_inds = [no in L_enable_nos for no in feat_nos]
        feat_nos = feat_nos[X_inds]

        idxdata = np.vstack((feat_nos, ylabels_conf[idcs]))
        np.savetxt(args.idx_file, idxdata, delimiter=",", fmt="%.0f")

        # グラフ画像出力
        self._cbam = np.array(confBAttM)
        self._zbam = np.array(zBAttM)
        logger.info("Finished.")

        pass

    def reset_model(self, args, logger, Cfg):
        # BAttM モデルの初期化
        # 状態変数の初期化
        Cfg.updatePatterns(Cfg.traffic_pat)
        # BAM モデル初期化
        Cfg.updateModel(Cfg.BAttM_model)

    def get_norm_name(self, name):
        """ 正規化後CSVファイル名を取得する """
        dirname = os.path.dirname(name)
        filename = os.path.basename(name)
        body, ext = os.path.splitext(filename)
        norname = "{}_norm{}{}".format(body, os.path.extsep, ext.strip('.'))
        return os.path.join(dirname, norname)

    def update_average(self, args, logger):
        # train(Xfile),対応ラベル読込
        N, X = DataSets(args.Xfile, logger).get()
        labels = YLabels(args.ylabel_name, logger)
        # 1以上のラベルのみアトラクタを構成する要素として扱う
        lset = {s if s >= 1 else None for s in set(labels.getValues())}
        lset.discard(None)
        ylbl_vals = sorted(lset)

        # ラベル種毎にXの平均を取得して配列化
        avgs = []
        for ylbl in ylbl_vals:
            ind_list = labels.getIndices(ylbl, N)
            avg = X[ind_list].mean(axis=0)
            avgs.append(avg)
        np_avg = np.array(avgs, dtype=float).T

        # 対応するylabelを１行目に追加して、平均値ファイルを上書き
        lbl_avg = np.vstack((ylbl_vals, np_avg))
        np.savetxt(args.AVGfile, lbl_avg, delimiter=',')

    def normalize_all(self, args, logger):
        """ 全観測刺激を取得し、正規化CSVを出力する """
        if args.normalize <= 0:
            return

        # AVGデータの読込
        AVG = np.loadtxt(args.AVGfile, dtype='float', delimiter=',', skiprows=0).T
        if len(AVG.shape) == 1:
            AVG = np.array([AVG])
        AVG_i = AVG[:, 0].astype(int)
        AVG = AVG[:, 1:]
        logger.info("Loaded {} shape category patterns from {}.".format(AVG.shape, args.AVGfile))
        # test, AVGの正規化

        # normalize 0:正規化なし、1:正規化パラメタあれば参照、2:新規正規化
        logger.info("Normalize mode %d" % args.normalize)

        normer = MinMaxNormalizer()
        # normalize >= 2または統計ファイルが無い場合は新たに正規化する.
        upd = not os.path.exists(args.min_max_all) or args.normalize > 1
        files = {"min_max_all": args.min_max_all}
        if upd and not args.train_only:
            logger.warn("no normalization parameter (Probably test without train)")
        if not upd:
            normer.load_stat(files)

        if args.train_only:
            # TRAIN正規化
            N, X = DataSets(args.Xfile, logger).get()
            logger.info("Loaded {} items in stimulus from {}.".format(X.shape, args.Xfile))
            X_norm = normer.normalize_feature_val(
                X, ylabel_name=None, update=upd, clip=True)
            # 統計ファイル出力
            if upd:
                normer.save_stat(files)
            X_out = np.vstack((N.T, X_norm.T))
            np.savetxt(self.get_norm_name(args.Xfile), X_out, delimiter=",")
            self._X = X_norm

            # Validation正規化
            N_valid, X_valid = DataSets(args.valid_file, logger).get()
            X_valid_norm = normer.normalize_feature_val(
                X_valid, ylabel_name=None, update=False, clip=True)
            X_valid_out = np.vstack((N_valid.T, X_valid_norm.T))
            np.savetxt(self.get_norm_name(args.valid_file), X_valid_out, delimiter=",")
            self._X_valid = X_valid_norm
        else:
            # TEST正規化 (正規化パラメタupdateなし)
            N_test, X_test = DataSets(args.test_file, logger).get()
            logger.info("Loaded {} items in stimulus from {}.".format(X_test.shape, args.test_file))
            X_test_norm = normer.normalize_feature_val(
                X_test, ylabel_name=None, update=False, clip=True)
            X_test_out = np.vstack((N_test.T, X_test_norm.T))
            np.savetxt(self.get_norm_name(args.test_file), X_test_out, delimiter=",")
            self._X_test = X_test_norm

        # AVGファイル処理 
        AVG_norm = normer.normalize_attractor(AVG, clip=True)
        AVG_out = np.vstack((AVG_i.T, AVG_norm.T))
        np.savetxt(self.get_norm_name(args.AVGfile), AVG_out, delimiter=",")
        self._AVG = AVG_norm
        return

    def run(self, args, logger):
        """ Main処理 """
        INPUT_DATA, PATTERN_DATA = self.init_data(args)

        # train指定であれば平均ファイルを作成
        if args.train_only:
            self.update_average(args, logger)
            # validationが無い場合、trainファイルからコピー
            if not os.path.exists(args.valid_file):
                shutil.copyfile(args.Xfile, args.valid_file)
                shutil.copyfile(args.ylabel_name, args.vlabel_name)

        # X.csv, AVG.csv すべて読み込み、正規化する
        self.normalize_all(args, logger)

        cat_patterns = self.init_cat_patterns(args, logger)

        input_name = args.valid_file if args.train_only else args.test_file

        # 全観測刺激を取得する. 
        X = self.get_stimulus(args, logger, input_name)
        if len(X) == 0:
            emsg = "No avaliable data for BAM stimulus." + \
                " check feature vector csv, and label definition csv."
            return emsg

        Cfg = self.init_configs(args, logger, cat_patterns)

        # BAttM モデルを作成する.
        if not Cfg.restored:
            logger.debug("Making BAttM model...")
            bam = BAttM(logger)

            #  # 観測関数 obs を作成する
            # Python では変数の組のみを保存する
            obs = bam.gen_observation(
                Cfg.traffic_pat, 
                Cfg.state_list, 
                Cfg.BAttM.r)

            BAttM_model  = bam.makeModel_PI(
                Cfg.state_list, 
                Cfg.BAttM.rambda, 
                obs, 
                Cfg.BAttM.q, 
                Cfg.s)

            #   # 設定情報に(再)作成した BAttM モデルを設定する.
            Cfg.updateModel(BAttM_model)
        else:
            pass

        # 刺激が無い場合は、モデル構築後に終了する. 
        if X is None:
            logger.info("X(stimulus) is not exists. STOP BY DCM before BAttM.estimation")
            return 

        if args.cli_mode > 0:
            BAttM_res = list()

        # 推定する. 
        logger.debug("BAttM estimation for %d steps" % len(X))

        # for(i in 1:length(X)){
        for i in np.arange(0, len(X)):
            for rpt in np.arange(0, args.repeat_cnt):
                logger.debug("Step#%d repeat %d ..." % (i, rpt))
                self.test1step(args, logger, bam, Cfg, i, X, BAttM_res)
            if args.reset_every > 0:
                logger.debug("Resetting model every input data..")
                self.reset_model(args, logger, Cfg)

        if args.cli_mode > 0 and args.no_out == 0:
            self.save_results(args, logger, bam, Cfg, X, BAttM_res)
        else:
            logger.info("Finished (called from python).")
            Cfg.restored = True
        pass

if __name__ == '__main__':
    est = BattmEstimator()
    ahelper = ArgsHelper()

    # 処理開始のデバッグメッセージを出力する. 
    args = ahelper.parser.parse_args()
    args = ahelper.format_args(args)

    # ロガーの作成
    logger = myLogger(args.level, args.logfile)
    logger.info("Started BAttM ver[%s] with args: %s" % (Python_BAttM_ver, args))
    est.run(args, logger)
    logger.close()
    pass

