import pandas as pd


def get_attributes_from_flow_list(flow_list):
    """
    Stateless function to extract attributes per T sec flow_list
    Input: List of flows from analyzer.py

    Currently using IQR for outlier detection - replace by better stats method
    Input to IQR: [ num_dst_ports, num_conns, src_tx, dst_tx ]
    - number of different unique dst ports being used in batch (groupby: srcip,dstip)
    - number of connections (packets) in batch
    - total traffic sent (uploaded)
    - total traffic received (downloaded)

    Results: Number of alerts generated
    - [num_dst_ports, num_conns]: 65752
    - [num_dst_ports]: 14366
    - [num_conns]: 60617
    - [num_dst_ports, num_conns, src_tx, dst_tx]: 76492

    return: zip(src_ip_list, dst_ip_list) iterable of outliers

    TODO: Make it state-full across data
        - remember previous open dst_ports and previous open flows that haven't been closed
        - a check every t=10 min to remove old entries from cache
    """
    df_flow = pd.DataFrame(flow_list)
    df_flow['src_ip'] = df_flow['src_ip'].apply(str)
    df_flow['dst_ip'] = df_flow['dst_ip'].apply(str)
    # print("len flow_list:", len(flow_list))
    # print("len df_flow:", len(df_flow))

    gp = df_flow.groupby(['src_ip', 'dst_ip'])
    df_ip = gp['src_tx', 'dst_tx'].sum()
    df_ip['num_conns'] = gp['state'].count()
    df_ip['num_dst_ports'] = gp['dst_port'].unique().apply(lambda x: len(x))
    df_ip['dst_ports_open'] = gp['dst_port'].unique()

    # TODO: dict of (src_ip, dst_ip) : [list of open ports]
    # dst_open_ports_current = df_ip['dst_ports_open'].to_dict()
    # dst_open_ports_current_timestamp = df_ip.iloc[0, 'ts']

    def find_iqr(ds):
        q1 = ds.quantile(0.25)
        q3 = ds.quantile(0.75)
        delta = (q3 - q1)
        # print(delta)
        inner_fence = (q1 - delta * 1.5, q3 + delta * 1.5)
        outer_fence = (q1 - delta * 3, q3 + delta * 3)
        return outer_fence

    outlier_features = ['num_dst_ports']    # , 'num_conns', 'dst_tx', 'src_tx']

    def outlier_detection(df):
        df_list = []
        for feature in outlier_features:
            l, u = find_iqr(df[feature])
            df3 = df[(df[feature] < l) | (df[feature] > u)][feature]
            # print(df3)
            if len(df_list) == 0:
                df_list = df3
            else:
                df_list = df_list.append(df3)
        df_list = df_list.reset_index()
        df_list[0] = 0
        df_list = df_list.drop_duplicates()
        src_list = list(df_list['src_ip'])
        dst_list = list(df_list['dst_ip'])
        return zip(src_list, dst_list)

    return outlier_detection(df_ip)
