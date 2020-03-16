#!/usr/bin/env Rscript
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
#    Class: Estimator
#     Role: Estimator Class.
# -------------------------------------------------------------------
#
getFilePath <- function() {
  initial.options <- commandArgs(trailingOnly = FALSE)
  file.arg.name <- "--file="
  script.name <- sub(file.arg.name, "", initial.options[grep(file.arg.name, initial.options)])
  script.basename <- dirname(script.name)
  other.name <- file.path(script.basename, "trace_debug.R")
  #print(paste("Sourcing",other.name,"from",script.name))
  dirname(other.name)
}

# バージョン情報.
R_BAttM_ver = "2019.04.02.1145"
run_in_python = FALSE

pwd = getFilePath()

# カレントディレクトリ配下の他ソースを読み込む.
source(paste(pwd, "/trace_debug.R", sep=""))
source(paste(pwd, "/Configs.R", sep=""))
source(paste(pwd, "/BAttM4ts.R", sep=""))
source(paste(pwd, "/Utils.R", sep=""))

# 処理開始のデバッグメッセージを出力する. 
Dbg.info(sprintf("Started BAttM run_in_python[%d] pwd[%s] ver[%s]",
         run_in_python, pwd, R_BAttM_ver))

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
INPUT_DATA = "current_obs_stimulus.csv"
# 引数が指定されている場合は引数を観測刺激にする. 
args = commandArgs(trailingOnly=TRUE)[1]
if (!is.na(args)){
  INPUT_DATA = args
}

# カテゴリの平均値ファイル名を取得する.
# Python から呼び出されている場合は直接変数 cat_patterns を
# アサインするため、ファイルは使用しない. 
PATTERN_DATA = "current_traffic_pat.csv"
# 引数が指定されている場合は引数をパターンにする. 
args = commandArgs(trailingOnly=TRUE)[2]
if (!is.na(args)){
  PATTERN_DATA = args
}

# カテゴリの平均値データを Python 変数から取得する.
# Python からアサインされていない場合は CSV ファイルから取得する. 
# Python::assign cat_patterns
if (!exists("cat_patterns")) {
  Dbg.info("Reading category patterns from file...")
  cat_patterns = read.csv(PATTERN_DATA)
} else {
  Dbg.info(sprintf("Loaded %d rows and %d columns category info from pyper", nrow(cat_patterns), ncol(cat_patterns)))
}

# 以前の設定が保存されている場合はリストアする.
# 存在しない場合は新規に作成する.
# Python::assign Cfg (after first step)
if (!exists("Cfg")) {
  Cfg = Configs.restoreConfig(cat_patterns)
} else {
  Dbg.debug("Updating category patterns...")
  Cfg = Configs.updateCategoryPatterns(Cfg, cat_patterns)
}

# BAttM/UKF のパラメータを更新する. 
if (exists("Params")) {
  Dbg.debug("Updating parameters...")
  Cfg = Configs.updateParams(Cfg, Params)
}

# 全観測刺激を取得する. 
# Python::assign X
if (!exists("X")) {
  # CSV ファイルが存在する場合読み込む.
  if (file.exists(INPUT_DATA)) {
    Dbg.info(sprintf("Reading stimulus from %s...", INPUT_DATA))
    X = read.csv(INPUT_DATA)
    Dbg.info(sprintf("Loaded %d items in stimulus file.", length(X)))
  }
} else {
  Dbg.info(sprintf("Loaded %d items in stimulus from pyper.", length(X)))
}

# BAttM モデルを作成する.
if (!Cfg$restored) {
  Dbg.info("Making BAttM model...")
  # 観測関数 obs を作成する
  obs = BAttM.gen_observation(Cfg$traffic_pat, Cfg$state_list, Cfg$BAttM.r)
  BAttM_model = BAttM.makeModel_PI(
    Cfg$state_list, Cfg$BAttM.rambda, obs, Cfg$BAttM.q, Cfg$s)
  # 設定情報に(再)作成した BAttM モデルを設定する.
  Cfg = Configs.updateBamModel(Cfg, BAttM_model)
} else {
  # 初回ではなく、再呼び出し (restored) の場合は BAttM モデルを作成しない.
}

# 刺激が無い場合は、モデル構築後に終了する. 
if (!exists("X")) {
  stop("X(stimulus) is not exists. STOP BY DCM before BAttM.estimation") 
}

# Python 呼び出しではない場合、結果リスト変数を作成する.
if (!run_in_python) {
  # 100回観測 (0 < t <= 100) を行ったとすれば、長さ100のリスト。
  # UKF の推定結果を格納したリスト。
  # 各推定結果には estx と estP が含まれる。
  BAttM_res = list() # BAttM_res はオンラインでは不要.
}

# 推定する. 
#Dbg.info("Estimating...")
Dbg.info(sprintf("BAttM estimation for %d steps",length(X)))
for(i in 1:length(X)){
  # 初回以外で Python 呼び出しではない場合、設定情報をリストアする.
  if ((i > 1) && !run_in_python) {
    Cfg = Configs.restoreConfig(cat_patterns)
  }

  # 基本的には BAttM モデルの 1step に入力する刺激X_iをオンライン入力にすれば良い.
  Dbg.info(sprintf("BAttM.estimation.1step for %d step", i))
  # 1 ステップ毎に推定する. 
  # Z_{t-1}、観測刺激 X_t、現在の BAttM モデルを入力として
  # 時刻 t における推定 x_t 及び推定 P_t を求める
  next_est = BAttM.estimation.1step(Cfg$BAttM_model, X[[i]], Cfg$estz, Cfg$Pz)
  # t-1 の estz/Pz 変数を t 時点の値で上書きする
  # estx: 意思決定状態 z の期待値
  Cfg$estz = next_est$estx
  # estP: 意思決定状態 z の共分散行列
  Cfg$Pz = next_est$estP
  # 結果をリストに格納する
  if (!run_in_python) {
    BAttM_res[[i]] = next_est
  }

  # 結果を正規化する.
  prob_stat <- rep(0, length(Cfg$state_list))
  # 全カテゴリに対して確率を計算(正規化)する
  for(i in 1:length(Cfg$state_list)){
    tmp = Cfg$state_list[[i]]
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
    prob_stat[i] <- dmvnorm(tmp, next_est$estx, next_est$estP)
  }

  if (!run_in_python) {
    # 結果をファイルに追記する. 
    Utils.WriteResult(prob_stat)
    # 状態を保存する. 
    Configs.saveConfig(Cfg)
  }
}

# Python::get prob_stat
# Python::get Cfg

# Python 呼び出しではない場合、結果を CSV ファイルに出力する. 
if (!run_in_python) {
  # 状態を保存
  Configs.saveConfig(Cfg)

  Dbg.info("Formatting results...")
  confBAttM = NULL
  # 全カテゴリの確信度を格納する
  for(i in 1:length(Cfg$state_list)){
    tmp = Cfg$state_list[[i]]
    # estx/estP をもとにして確信度を計算する
    # lapply でリスト（各step毎）に関数を適用して、結果をリストに格納する
    conf_res = lapply(BAttM_res, function(x){BAttM.confidence(tmp,x$estx,x$estP)})
    # unlist でリストを一つのベクトルにまとめ、行列に連結する
    confBAttM = cbind(confBAttM, unlist(conf_res))
  }

  zBAttM = NULL
  pBAttM = NULL
  for(i in 1:length(X)){
    # 観測刺激の予測値 z の結果を行列に格納する
    zBAttM = cbind(zBAttM,BAttM_res[[i]]$estx)
    # 各列の合計値を求め、アトラクタ毎の共分散の和を計算する
    pBAttM = rbind(pBAttM,apply(BAttM_res[[i]]$estP*BAttM_res[[i]]$estP,2,sum))
  }

  iBAttM = NULL
  for(i in 1:length(X)){
    if (max(confBAttM[i,]) > 0.0) # 0.0001
      iBAttM = cbind(iBAttM,which.max(confBAttM[i,]))
    else
      iBAttM = cbind(iBAttM, -1)
  }

  # 結果を出力
  Dbg.info("Writing CSVs...")
  # 確信度: 各時刻、各カテゴリの確信度を格納した行列 
  # 値範囲は0-1 ではない
  write.csv(confBAttM, "prob_static.csv", row.names = F)
  # z空間の位置 (観測刺激の予測値に等しい)
  write.csv(zBAttM, "zBAttM.csv", row.names = F)
  # アトラクタ毎の共分散和
  write.csv(pBAttM, "pBAttM.csv", row.names = F)
  # 最大確信度のIndex値
  # install.packages('ramify')

  write.csv(iBAttM, "iBAttM.csv", row.names = F)

  Dbg.info("Finished.")
} else {
  Dbg.info("Finished (called from python).")
  Cfg$restored = TRUE
}


