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
#    Class: Configs
#     Role: Configuration Variable Class.
# -------------------------------------------------------------------
# 
# 
source(paste(pwd, "/trace_debug.R", sep=""))

CONFIG_FILE_NAME = "serialized.rds"

# estz/Pz 及び BAttM モデルなど、1 step 毎の動作を実現するために必要な
# 情報を保持する Configs リスト変数を作成する。
Configs.makeConfig = function(cat_pat) {
  Dbg.fbegin(match.call()[[1]])
  config = list()

  # ----------------------------------------
  # バージョン情報
  config$version <- 0.1
  # 復元有無
  config$restored <- FALSE
  # デバッグレベル
  config$debug_level <- DEBUG

  # ----------------------------------------
  # 定数群
  #
  # センサの不確実性 (sensory uncertainty)
  # SQRT(データ分散) で良いが、事前には分からない
  config$s <- 0.100
  
  # Bitzer 論文の式(10)の r (シグモイド関数パラメータ)
  config$BAttM.r <- 2.7
  
  # Zの更新式のシグモイド関数の傾き
  # 論文中 beta: シグモイド関数パラメータ
  config$BAttM.rambda <- 1.4
  
  # ホップフィールドの更新式のノイズ項の分散
  # 論文の q: dynamics uncertainty
  config$BAttM.q <- 0.4898979
  
  # UKF パラメータ
  # 平均の状態値の周りのシグマ
  # 標準偏差ポイントパラメータ
  config$UKF.alpha <- 0.01
  
  # UKF パラメータ
  # 状態の分布の事前情報
  # ガウス分布の場合、beta = 2 が最適。
  config$UKF.beta <- 2
  
  # UKF パラメータ
  # 2 番目のスケーリング パラメーター
  # 通常は 0 に設定される。
  # 値がさいほど、シグマ ポイントは平均の状態に近くなる。
  # 広がりは kappa の平方根に比例する。
  config$UKF.kappa <- 0

  config <- Configs.updateCategoryPatterns(config, cat_pat)

  Dbg.fend(match.call()[[1]])
  return(config)
}

# BAttM/UKF のパラメータを更新する. 
Configs.updateParams = function(config, params) {
  # デバッグレベル
  config$debug_level <- Params$debug_level
  Dbg.LEVEL <<- config$debug_level

  # 定数群
  config$s            <- params$s
  config$BAttM.r      <- params$BAttM.r
  config$BAttM.rambda <- params$BAttM.rambda
  config$BAttM.q      <- params$BAttM.q
  config$UKF.alpha    <- params$UKF.alpha
  config$UKF.beta     <- params$UKF.beta
  config$UKF.kappa    <- params$UKF.kappa

  return(config)
}

# カテゴリの平均値情報を更新する.
Configs.updateCategoryPatterns = function(config, cat_pat) {
  # ----------------------------------------
  # カテゴリパターン数で値が変化する変数群
  #
  # 特徴量の次元数
  # 型：64個の特徴量があるとすれば、64 が入る
  config$dim_x <- length(cat_pat[[1]])
  # カテゴリ数
  config$num_a <- length(cat_pat)
  # Z 空間の次元数
  # 最低でも「カテゴリ数 +1」を指定する必要がある
  config$dim_z <- config$num_a
  # カテゴリパターン、各カテゴリの代表特徴量値. 
  # 型：１行目はカテゴリ番号、２行目以降は特徴量が入る。
  # カテゴリの特徴量平均値などが入る。
  # 例: データ例(カテゴリ数=3、特徴量の軸数=4の場合)：
  #    [[1,    2,    3],
  #    [0.00, 0.03, 0.04],
  #    [0.00, 0.12, 0.15],
  #    [0.03, 0.22, 0.17],
  #    [0.25, 0.22, 0.22]]
  config$traffic_pat <- cat_pat

  # ----------------------------------------
  # アトラクター(状態)リストの作成
  # 型: カテゴリ数 × カテゴリ数の行列
  # 例：初期値(カテゴリ数=3の場合)：
  # [[+1, -1, -1],
  #  [-1, +1, -1],
  #  [-1, -1, +1]]
  config$state_list = list()
  for(i in 1:config$num_a){
    config$state_list[[i]] <- rep(-1, config$dim_z)
    config$state_list[[c(i,i)]] <- 1
    Dbg.debug(sprintf("i=%d, dim_z=%d, num_a=%d",
            i, config$dim_z, config$num_a))
  }

  return(config)
}

# BAttM モデルを更新する. 
Configs.updateBamModel = function(config, BAttM_model) {
  config$BAttM_model <- BAttM_model
  config$estz <- BAttM_model$x0
  config$Pz <- BAttM_model$P0
  return(config)
}

# 現在の BAttM モデルを含めた設定情報をファイル保存(シリアライズ)する.
Configs.saveConfig = function(cfg) {
  Dbg.info(sprintf("Saving configuration file to %s...", CONFIG_FILE_NAME))
  #saveRDS(cfg, CONFIG_FILE_NAME, refhook = xmlSerializeHook)
  saveRDS(cfg, CONFIG_FILE_NAME, ascii = FALSE, compress = TRUE)
}

# ファイルに保存されている BAttM モデルを復元(デシリアライズ)する. 
Configs.restoreConfig = function(cat_pat) {
  if (file.exists(CONFIG_FILE_NAME)) {
    Dbg.info(sprintf("Restoring configuration file from %s...", CONFIG_FILE_NAME))
    #cfg = readRDS(CONFIG_FILE_NAME, refhook = xmlDeserializeHook)
    cfg <- readRDS(CONFIG_FILE_NAME)
    cfg$restored <- TRUE
    Dbg.LEVEL <<- cfg$debug_level
    #Dbg.debug(str(cfg))
  } else {
    Dbg.info("Creating new configuration with given category patterns...")
    cfg <- Configs.makeConfig(cat_pat)
  }

  return(cfg)
}

