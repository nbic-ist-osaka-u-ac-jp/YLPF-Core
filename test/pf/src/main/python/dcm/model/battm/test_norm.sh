#!/bin/sh -x
#
# -------------------------------------------------------------------
# -*- coding: utf-8 -*-
#
# Copyright (C) 2020 NEC Brain Inspired Computing
#                    Research Alliance Laboratories
#                    All Rights Reserved.
# -------------------------------------------------------------------

BASE_DIR=../../../../../../../../pf/src/main/python/dcm/model/battm

python3 $BASE_DIR/bam_py/est.py \
	--test_file test.csv \
	--train_only 0 \
	--AVGfile average.csv \
	--base_dir . \
	--logfile 'log.txt' \
	--bam_s 0.1 \
	--bam_r 2.7 \
	--bam_rambda 1.4 \
	--bam_q 0.489898 \
	--ukf_alpha 0.01 \
	--ukf_beta 2.0 \
	--ukf_kappa 0.0 \
	--reset_every 0 \
	--repeat_cnt 1 \
	--select_data ':' \
	--select_feat ':' \
	--normalize 1 \
	--level 20
