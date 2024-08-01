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

_POPULAR_PORTS = {
    80: 'http',
    8080: 'http',
    443: 'ssl',
    22: 'ssh',
    53: 'dns',
    123: 'ntp',
    143: 'imap',
    993: 'imap-ssl',
}