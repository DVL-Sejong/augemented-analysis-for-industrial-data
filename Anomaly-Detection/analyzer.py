#!/usr/bin/python
import collections
import csv
import datetime
import ipaddr
import sys
import numpy as np

from pandas_analysis import get_attributes_from_flow_list

# from blacklist_update import Blacklist

_FLOW_FIELDS = [
    "ts",
    "ip_protocol",
    "state",
    "src_ip",
    "src_port",
    "dst_ip",
    "dst_port",
    "src_tx",
    "dst_tx",
]
