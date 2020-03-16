#!/bin/sh -x
#
# -------------------------------------------------------------------
#
# Copyright (C) 2020 NEC Brain Inspired Computing
#                    Research Alliance Laboratories
#                    All Rights Reserved.
# -------------------------------------------------------------------

find . -type f -name '.DS_Store' | sed -e 's/^/rm -f /;' | sh -x
find . -type f -name '*.pyc' | sed -e 's/^/rm -f /;' | sh -x
find . -type f -name '*~' | sed -e 's/^/rm -f /;' | sh -x
find . -type d -name '__pycache__' | sed -e 's/^/rm -fr /;' | sh -x
