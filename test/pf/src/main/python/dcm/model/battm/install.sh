#!/bin/sh -x
#
# -------------------------------------------------------------------
# -*- coding: utf-8 -*-
#
# Copyright (C) 2020 NEC Brain Inspired Computing
#                    Research Alliance Laboratories
#                    All Rights Reserved.
# -------------------------------------------------------------------

X=`grep Raspbian /etc/*-release`

if [ "x$X" != "x" ] ; then
    sudo apt-get install llvm
fi

sudo -H pip3 install scikit-learn
sudo -H pip3 install numpy
sudo -H pip3 install scipy

sync
