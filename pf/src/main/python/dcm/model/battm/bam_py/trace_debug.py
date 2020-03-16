#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Component: YLPF
# Package  : battm
# Name     : Logger
# Author   : NBIC
# License  : BSD 2-Clause
# -------------------------------------------------------------------
#
# Copyright (C) 2019 NEC Brain Inspired Computing
#                    Research Alliance Laboratories
#                    All Rights Reserved.
# -------------------------------------------------------------------

import logging
from datetime import datetime

class CustomLogger(object):
    def __init__(self, tag):
        self._tag = tag
        self._level = logging.DEBUG
        self._fname = None
        self._q = []
        pass

    def setLevel(self, lvl):
        self._level = lvl

    def setFilename(self, name):
        self._fname = name

    def flush(self):
        if self._fname:
            with open(self._fname, mode="a") as f:
                for l in self._q:
                    f.write(l)
        pass

    def close(self):
        self.flush()
        self._q.clear()

    def outstring(self, lvl, msg):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
        log = "{} {} {} :{}\n".format(now, lvl, self._tag, msg.strip())
        self._q.append(log)

    def error(self, msg):
        if logging.ERROR >= self._level:
            self.outstring("ERROR", msg)

    def warn(self, msg):
        if logging.WARN >= self._level:
            self.outstring("WARN", msg)

    def info(self, msg):
        if logging.INFO >= self._level:
            self.outstring("INFO", msg)

    def debug(self, msg):
        if logging.DEBUG >= self._level:
            self.outstring("DEBUG", msg)

    def trace(self, msg):
        if logging.NOTSET >= self._level:
            self.outstring("TRACE", msg)


def myLogger(lvl=logging.DEBUG, logfile=None):
    logger = CustomLogger("BAttM")
    logger.setLevel(lvl)
    logger.setFilename(logfile)
    return logger

