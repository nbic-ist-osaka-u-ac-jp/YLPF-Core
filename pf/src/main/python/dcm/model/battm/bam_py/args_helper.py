#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Component: YLPF
# Package  : battm
# Name     : ArgsHelper
# Author   : NBIC
# License  : BSD 2-Clause
# -------------------------------------------------------------------
#
# Copyright (C) 2018 NEC Brain Inspired Computing
#                    Research Alliance Laboratories
#                    All Rights Reserved.
# -------------------------------------------------------------------

import argparse
import json
import logging
import os
from args_members import ArgsMembers

class ArgsHelper(object):
    def __init__(self):
        self.parser = None
        self.make_arg_parser()
        pass

    def make_arg_parser(self):
        """ 引数パーサーを作成する """
        parser = argparse.ArgumentParser()
        parser.add_argument('--Xfile',       type=str,   default="train.csv", help='Stimulus File')
        parser.add_argument('--train_only',  type=int,   default=1, help='Train Only')
        parser.add_argument('--AVGfile',     type=str,   default="average.csv", help='Average File')
        parser.add_argument('--cli_mode',    type=int,   default=1, help='CLI Mode')
        parser.add_argument('--no_out',      type=int,   default=0, help='Fake CSV Write')
        parser.add_argument('--save_cfg',    type=int,   default=0, help='Save/Restore Config.')
        parser.add_argument('--pkl_name',    type=str,   default="", help='シリアライズ設定ファイル')
        parser.add_argument('--conf_file',   type=str,   default="cBAttM.csv", help='Confidence')
        parser.add_argument('--z_file',      type=str,   default="zBAttM.csv", help='Z file')
        parser.add_argument('--p_file',      type=str,   default="pBAttM.csv", help='p file')
        parser.add_argument('--idx_file',    type=str,   default="iBAttM.csv", help='Index file')
        parser.add_argument('--level',       type=int,   default=logging.INFO, help='logger level (Debug:10 Info:20 Warn:30 Error:40 Fatal:50)')
        parser.add_argument('--base_dir',    type=str,   default="./", help='base directory')
        parser.add_argument('--result_file', type=str,   default="", help='confidence file')
        parser.add_argument('--bam_r',       type=float, default=3.0000000, help='r of equation 10 in Bitzer')
        parser.add_argument('--bam_rambda',  type=float, default=1.0000000, help='beta in ')
        parser.add_argument('--bam_q',       type=float, default=0.4898979, help='dynamics uncertainty')
        parser.add_argument('--bam_s',       type=float, default=0.4000000, help='sensory uncertainty')
        parser.add_argument('--ukf_alpha',   type=float, default=0.0100000, help='UKF alpha')
        parser.add_argument('--ukf_beta',    type=float, default=2.0000000, help='UKF beta')
        parser.add_argument('--ukf_kappa',   type=float, default=0.0000000, help='UKF kappa')
        parser.add_argument('--repeat_cnt',  type=int,   default=10, help='repeat count per sample')
        parser.add_argument('--logfile',     type=str,   default="log.txt", help='log filename')
        parser.add_argument('--reset_every', type=int,   default=1, help='reset model every input data')
        parser.add_argument('--select_data', type=str,   default=':', help='select input data with array indices')
        parser.add_argument('--select_feat', type=str,   default='', help='select feature vectors with array indices')
        parser.add_argument('--json_file',   type=str,   default='cfg.json', help='used configuration file')
        parser.add_argument('--select_attr', type=str,   default='0,1', help='array indices of selected attractors')
        parser.add_argument('--ylabel_name', type=str,   default='ylabel.csv', help='y-label filename')
        parser.add_argument('--timestamp',   type=str,   default='', help='timestamp for start time')
        parser.add_argument('--normalize',   type=int,   default=1, help='normalize mode. 0:none, 1:refer, 2:new')
        parser.add_argument('--min_max_all',   type=str,   default='min_max_all.csv', help='normalization parameter file')
        parser.add_argument('--test_file',   type=str,   default='test.csv', help='test input data file')
        parser.add_argument('--valid_file',  type=str,   default='validation.csv', help='validation input data file')
        parser.add_argument('--vlabel_name', type=str,   default='vlabel.csv', help='validation input label filename')

        self.parser = parser
        return parser

    def format_args(self, args):
        """ 引数を整形する """
        # 初期値投入
        if args.Xfile is "":
            args.Xfile = "current_obs_stimulus.csv"
        if args.AVGfile is "":
            args.AVGfile = "current_traffic_pat.csv"

        # ファイル名にベースディレクトリを付与する
        if not args.base_dir in args.Xfile:
            args.Xfile       = os.path.join(args.base_dir, args.Xfile)
        if not args.base_dir in args.AVGfile:
            args.AVGfile     = os.path.join(args.base_dir, args.AVGfile)
        if not args.base_dir in args.conf_file:
            args.conf_file   = os.path.join(args.base_dir, args.conf_file)
        if not args.base_dir in args.z_file:
            args.z_file      = os.path.join(args.base_dir, args.z_file)
        if not args.base_dir in args.p_file:
            args.p_file      = os.path.join(args.base_dir, args.p_file)
        if not args.base_dir in args.idx_file:
            args.idx_file    = os.path.join(args.base_dir, args.idx_file)

        if args.pkl_name != "" and not args.base_dir in args.pkl_name:
            args.pkl_name = os.path.join(args.base_dir, args.pkl_name)
        if args.result_file != "" and not args.base_dir in args.result_file:
            args.result_file = os.path.join(args.base_dir, args.result_file)
        if args.logfile != "" and not args.base_dir in args.logfile:
            args.logfile = os.path.join(args.base_dir, args.logfile)
        if args.json_file != "" and not args.base_dir in args.json_file:
            args.json_file = os.path.join(args.base_dir, args.json_file)
        if args.ylabel_name != "" and not args.base_dir in args.ylabel_name:
            args.ylabel_name = os.path.join(args.base_dir, args.ylabel_name)
        if args.min_max_all != "" and not args.base_dir in args.min_max_all:
            args.min_max_all = os.path.join(args.base_dir, args.min_max_all)
        if args.test_file != "" and not args.base_dir in args.test_file:
            args.test_file = os.path.join(args.base_dir, args.test_file)
        if args.valid_file != "" and not args.base_dir in args.valid_file:
            args.valid_file = os.path.join(args.base_dir, args.valid_file)
        if args.vlabel_name != "" and not args.base_dir in args.vlabel_name:
            args.vlabel_name = os.path.join(args.base_dir, args.vlabel_name)

        return args

    def get_default_args(self):
        if self.parser is None:
            self.parser = self.make_arg_parser()
        args = ArgsMembers(self.parser.parse_args([]))
        return args

    def json2args(self, msg):
        args = self.get_default_args()

        dic = json.loads(msg)

        for key in dic.keys():
            #print("args.__dict__[{}] = dic[{}] = {}".format(key, key, dic[key]))
            args.__dict__[key] = dic[key]

        return args

