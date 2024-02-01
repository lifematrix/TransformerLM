# -*- coding: utf-8 -*-

import yaml
from datetime import datetime


class CommUtils:
    @classmethod
    def load_yaml_config(cls, config_file):
        """Read and parse config file
        """
        with open(config_file, "r") as f:
            config_txt = f.read()
            config = yaml.load(config_txt, Loader=yaml.FullLoader)
        return config, config_txt

    @classmethod
    def now_str(cls):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
