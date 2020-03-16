#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Component: YLPF
# Package  : battm
# Name     : ArgsMembers
# Author   : NBIC
# License  : BSD 2-Clause
# -------------------------------------------------------------------
#
# Copyright (C) 2018 NEC Brain Inspired Computing
#                    Research Alliance Laboratories
#                    All Rights Reserved.
# -------------------------------------------------------------------

import json

class ArgsMembers(object):
    """ CLI と互換性がある引数クラス """
    def __init__(self, args):
        self.Xfile = args.Xfile
        self.train_only = args.train_only
        self.AVGfile = args.AVGfile
        self.cli_mode = args.cli_mode
        self.no_out = args.no_out
        self.save_cfg = args.save_cfg
        self.pkl_name = args.pkl_name
        self.conf_file = args.conf_file
        self.z_file = args.z_file
        self.p_file = args.p_file
        self.idx_file = args.idx_file
        self.level = args.level
        self.base_dir = args.base_dir
        self.result_file = args.result_file
        self.bam_r = args.bam_r
        self.bam_rambda = args.bam_rambda
        self.bam_q = args.bam_q
        self.bam_s = args.bam_s
        self.ukf_alpha = args.ukf_alpha
        self.ukf_beta = args.ukf_beta
        self.ukf_kappa = args.ukf_kappa
        self.repeat_cnt = args.repeat_cnt
        self.logfile = args.logfile
        self.reset_every = args.reset_every
        self.select_data = args.select_data
        self.select_feat = args.select_feat
        self.json_file = args.json_file
        self.select_attr = args.select_attr
        self.ylabel_name = args.ylabel_name
        self.timestamp = args.timestamp
        self.normalize = args.normalize
        self.min_max_all = args.min_max_all
        self.test_file = args.test_file
        self.valid_file = args.valid_file
        self.vlabel_name = args.vlabel_name

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__)

    def fromJSON(self, msg):
        dic = json.loads(msg)
        for key in dic.keys():
            self.__dict__[key] = dic[key]

        return self

