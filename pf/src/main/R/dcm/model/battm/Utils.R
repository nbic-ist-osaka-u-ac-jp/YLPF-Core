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
#    Class: Utils
#     Role: Utilitiy Class.
# -------------------------------------------------------------------
# 
# 
# 結果(確信度)をファイルに出力する. 
Utils.WriteResult = function(x) {
  # CSV用データを整形する
  result <- NULL
  for (i in 1:(length(x) -1)) {
    result <- paste(result, x[i], sep="")
    result <- paste(result, ",", sep="")
  }
  result <- paste(result, x[length(x)], sep="")
  result <- paste(result, "\n", sep="")

  # 追記モードで開く
  out_file = file("result.csv", "a")
  writeLines(result, out_file, sep="")
  close(out_file)
}
