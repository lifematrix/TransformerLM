# -*- coding: utf-8 -*-
################################################################################
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################


import logging
import os
from logging.handlers import TimedRotatingFileHandler

#
#  The following color-log come from <https://gist.github.com/KurtJacobson/c87425ad8db411c73c6359933e5db9f9>
#
from copy import copy
from logging import Formatter


class LogUtils:
    class ColoredFormatter(Formatter):
        #
        # for shell color, refer to <https://misc.flogisoft.com/bash/tip_colors_and_formatting>
        #
        MAPPING = {
            'DEBUG': 35,  # Magenta
            'INFO': 32,  # green
            'WARNING': 33,  # yellow
            'ERROR': 31,  # red
            'CRITICAL': 27,  # reset blink
        }

        PREFIX = '\033['
        SUFFIX = '\033[0m'

        def __init__(self, patern, datefmt):
            self.datefmt = datefmt
            Formatter.__init__(self, patern, datefmt=datefmt)

        def format(self, record,):

            colored_record = copy(record)
            levelname = colored_record.levelname
            seq = self.MAPPING.get(levelname, 37) # default white
            colored_levelname = ('{0}{1}m{2}{3}') \
                .format(self.PREFIX, seq, levelname, self.SUFFIX)
            colored_record.levelname = colored_levelname

            # colored_record.msg = ('{0}{1}m{2}{3}') \
            #     .format(self.PREFIX, seq, record.msg, self.SUFFIX)
            return Formatter.format(self, colored_record)

    #
    # initialize log
    #
    @classmethod
    def initlog(cls, logpath="./log", logfname="app", add_pid=False, rank=None):
        # add logging to console
        print("init log, rank: %s" % rank)
        if add_pid:
            logfname = "%s.%s.log" % (logfname, os.getpid())
        else:
            logfname = "%s.log" % logfname

        if rank is not None:
            logfname += ".rank-%s" % rank


        colored_formatter = cls.ColoredFormatter("[%(asctime)s(%(relativeCreated)d)] %(levelname)s\t%(message)s", datefmt='%m/%d/%y %H:%M:%S')
        console_logger = logging.StreamHandler()
        console_logger.setFormatter(colored_formatter)
        if os.getenv('DEBUG_TO_CONSOLE', '0') == '1':
            console_logger.setLevel(logging.DEBUG)
        else:
            console_logger.setLevel(logging.INFO)
        # logging.getLogger().addHandler(console)
        logging.basicConfig(
            format='%(asctime)s(%(relativeCreated)d) - %(levelname)s %(filename)s(%(lineno)d) :: %(message)s',
            level=logging.DEBUG,
            handlers=[console_logger])
        # import os, datetime
        # logging.basicConfig(
        #      format='%(asctime)s(%(relativeCreated)d) - %(levelname)s %(filename)s(%(lineno)d) :: %(message)s',
        #      level=logging.DEBUG,
        # )
        timeHandler = TimedRotatingFileHandler(os.path.join(logpath, logfname), when="midnight", interval=1)
        timeHandler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s(%(relativeCreated)d) - %(levelname)s %(filename)s(%(lineno)d) :: %(message)s')
        timeHandler.setFormatter(formatter)
        timeHandler.suffix = "%Y%m%d"
        logging.getLogger().addHandler(timeHandler)

    @classmethod
    def default_init(cls):
        logging.basicConfig(
            format='%(asctime)s(%(relativeCreated)d) - %(levelname)s %(filename)s(%(lineno)d) :: %(message)s',
            level=logging.DEBUG)


initlog = LogUtils.initlog



