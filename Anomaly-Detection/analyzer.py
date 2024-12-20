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

_INTERESTING_PORTS = {
    0: 'reserved',
    81: 'Tor',
    82: 'Tor-control',
}

_INTERESTING_PORTS = {
    0: 'reserved',
    81: 'Tor',
    82: 'Tor-control',
}

class Flow(collections.namedtuple("Flow", _FLOW_FIELDS)):
    __slots__ = ()

    @staticmethod
    def from_csv(e):
        """
        Factory method.

        Construct Flow instances from a CSV-representation of a flow.
        """
        return Flow(ts=datetime.datetime.strptime(e[0], "%Y-%m-%d %H:%M:%S"),
                    ip_protocol=e[1],
                    state=e[2],
                    src_ip=ipaddr.IPAddress(e[3]),
                    src_port=int(e[4]),
                    dst_ip=ipaddr.IPAddress(e[5]),
                    dst_port=int(e[6]),
                    src_tx=int(e[7]),
                    dst_tx=int(e[8]))

_ALERT_FIELDS = [
    "name",
    "evidence",
]

Alert = collections.namedtuple("Alert", _ALERT_FIELDS)

class Analyzer(object):

    def __init__(self):
        self.__num_flows = 0
        self.__alerts = []

        self.__safe_ips = set()
        self.__load_blacklist()

        self.__port_stats = {}
        self.__ip_stats = {}
        self.__num_ports_average = 5

        self.__T = 10   # seconds to aggregate and load in memory
        self.__refresh_ports_cycle = 60     # cycles to refresh dst_ports
        self.__refresh_ports_counter = 0
        self.flow_list = []
    
    def __load_blacklist(self):
        with open('blacklist_ips.csv', 'r') as blacklistcsv:
            self.__blacklist = set(list(csv.reader(blacklistcsv))[0])
        print("load blacklist")
    
    def alert_basic_checks(self, flow):
        src_ip = flow.src_ip.exploded
        dst_ip = flow.dst_ip.exploded

        if flow.dst_port in _INTERESTING_PORTS:
            self.__alerts.append(Alert(name="Using interesting port number "+str(flow.dst_port),
                                       evidence=[flow]))
        
        if (src_ip is "0.0.0.0" and dst_ip is "255.255.255.255") or\
                (src_ip is not "0.0.0.0" and dst_ip is "255.255.255.255"):
            self.__alerts.append(Alert(name="Malformed DHCP or local gateway flow",
                                        evidence=[flow]))

        if flow.src_tx == 0 and flow.dst_tx == 0:
            pass

        if flow.src_ip.is_private and flow.dst_ip.is_private:
            pass

    def alert_ip_blacklist(self, flow):
        