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
#    Class: Dbg
#     Role: Debug Log Class.
# -------------------------------------------------------------------
# 
# 

# デバッグレベルの定義
TRACE   = 0
DEBUG   = -1
INFO    = -2
WARNING = -3
ERROR   = -4

# グローバル変数が未定義の場合、新規に宣言する
if (!exists("Dbg.LEVEL")) {
  Dbg.LEVEL = DEBUG
}

# 標準出力が未定義であれば TRUE に定義する
if (!exists("Dbg.stdout")) {
  Dbg.stdout = TRUE
}

# ファイル出力が未定義であれば TRUE に定義する
if (!exists("Dbg.fileout")) {
  Dbg.fileout = TRUE
}

# デバッグメッセージ出力関数を定義する
Dbg.outstring =function(xxx) {
  # 標準出力
  if (Dbg.stdout) {
    print(xxx)
  }

  # ファイルに出力
  if (Dbg.fileout) {
    # 日付時刻を整形する
    cur_time <- format(Sys.time(),"[%Y-%m-%d %H:%M:%S] ")
    # 追記モードでファイルを開き、ログ出力を追加する
    out <- file("/tmp/battm_classifier.log", "a")
    writeLines(paste(cur_time, xxx, "\n"), out, sep="")
    close(out)
  }
}

# エラー情報の出力関数を定義する
Dbg.error = function(xx) {
  if (Dbg.LEVEL >= ERROR) {
    Dbg.outstring(paste("ERROR:", xx))
  }
}

# 警告情報の出力関数を定義する
Dbg.warn = function(xx) {
  if (Dbg.LEVEL >= WARNING) {
    Dbg.outstring(paste(" WARN:", xx))
  }
}

# 通知情報の出力関数を定義する
Dbg.info = function(xx) {
  if (Dbg.LEVEL >= INFO) {
    Dbg.outstring(paste(" INFO:", xx))
  }
}

# デバッグ情報の出力関数を定義する
Dbg.debug = function(xx) {
  if (Dbg.LEVEL >= DEBUG) {
    Dbg.outstring(paste("DEBUG:", xx))
  }
}

# トレース情報の出力関数を定義する
Dbg.trace = function(xx) {
  if (Dbg.LEVEL >= TRACE) {
    Dbg.outstring(paste("TRACE:", xx))
  }
}

# 関数開始のトレース情報出力関数を定義する
Dbg.fbegin = function(xx) {
  if (Dbg.LEVEL >= TRACE) {
    indent_str = ""
    # スタックフレームを取得し、一つ上の関数名を取得する
    f_depth = sys.nframe() - 1
    for (i in 1:f_depth) {
      indent_str = paste(indent_str, ".")
    }
    Dbg.outstring(paste("FUNC:[", f_depth, "]", "Enter", indent_str, xx))
  }
}

# 関数終了のトレース情報出力関数を定義する
Dbg.fend = function(xx) {
  if (Dbg.LEVEL >= TRACE) {
    indent_str = ""
    # スタックフレームを取得し、一つ上の関数名を取得する
    f_depth = sys.nframe() - 1
    for (i in 1:f_depth) {
      indent_str = paste(indent_str, ".")
    }
    Dbg.outstring(paste("FUNC:[", f_depth, "]", "  End", indent_str, xx))
  }
}
