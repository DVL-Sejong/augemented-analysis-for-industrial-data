# This is New York Traffic Speed Data

The TMC maintains a map of traffic speed detectors throughout the City. The speed detector themselves belong to various city and state agencies. The Traffic Speeds Map is available on the DOT's website. This data feed contains 'real-time' traffic information from locations where DOT picks up sensor feeds within the five boroughs, mostly on major arterials and highways. DOT uses this information for emergency response and management. The metadata defines the fields available in this data feed and explains more about the data. This dataset is a result of scraping data from the DOT site and is not an official DOT product.

source : NYC Real Time Traffic Speed Data Feed(Archived) Five Minute Intervals
link : https://data.beta.nyc/dataset/nyc-real-time-traffic-speed-data-feed-archived/resource/2cebdd67-0b64-4753-b0a6-2751cb6e866f?inner_span=True

### Data Dictionary
|Column|Data Type|
|:---|:---|
|Id|numeric|
|Speed|numeric|
|TravelTime|numeric|
|Status|numeric|
|DataAsOf|timestamp|
|LinkId|numeric|

Duration of collected data : 4/1/2015 ~ 12/31/2022

### Link Data Dictionary
|Column|Data Type|
|:---|:---|
|LinkId|text|
|LinkPoints|text|
|EncodedPolyLine|text|
|EncodedPolyLineLvls|text|
|Transcom_id|text|
|Borough|text|
|linkName|text|
|Owner|text|