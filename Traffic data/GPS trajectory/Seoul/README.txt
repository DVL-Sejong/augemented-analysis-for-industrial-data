The dataset is collected by Yonsei University.
We deployed our mobility monitoring system, named LifeMap, to collect mobility data over two months in Seoul, Korea.

LifeMap used learning scheme proposed in following paper. Please refer this paper when you use our dataset.
* Yohan Chon, Elmurod Talipov, Hyojeong Shin, and Hojung Cha. 2011. Mobility prediction-based smartphone energy optimization for everyday location monitoring. In Proceedings of the 9th ACM Conference on Embedded Networked Sensor Systems (SenSys '11). ACM, New York, NY, USA, 82-95.

Visit our homepage for more information (http://lifemap.yonsei.ac.kr).

#########################################
How to extract mobility data from dataset
#########################################
We recommend that the use of stayTable for extracting mobility traces instead of the use of edgeTable.
We provided the example of JAVA source code to handle database file. The source code can be imported projects in eclipse.
See showMobilityTrace() in LifeMapDatabase.java
To run example code in Eclipse,
1. Copy our dataset and sqlitejdbc-v056.jar (http://www.zentus.com/sqlitejdbc/) into project paths.
2. Run eclipse and import project.
2. Set sqlitejdbc-v056.jar to library of projects.
3. Run java application in Main.java

####################################################
How to import database file into Android application
####################################################
1. Copy database file into your Android device.
2. download and install 'LifeMap' application from Android market (https://market.android.com/details?id=com.mobed.lifemap).
3. Click sensond tab, and open 'Menu' -> 'Manage Space' -> 'Load Database.'
4. Type the path of database file name (e.g., /sdcard/data/LifeMap_GS1.db).
5. Restart 'LifeMap' application.


######################
Desription of data set
######################
------------------------------------------------------------------------
| User + Sex + Age + Number of Place + Periods + Start Date + End Date |
------------------------------------------------------------------------
| GS1 + Male + 20s + 92 + 134 + 2011.3.8 + 2011.7.19 |
| GS2 + Male + 20s + 163 + 198 + 2011.5.2 + 2011.11.15 |
| GS3 + Male + 20s +  + 297 + 2010.11.17 + 2011.8.14 |
| GS4 + Female + 20s + 209 + 183 + 2011.5.7 + 2011.11.15 |
| GS7 + Male + 20s + 289 + 132 + 2011.3.10 + 2011.7.19 |
| GS8 + Male + 30s + 345 + 250 + 2011.3.10 + 2011.11.14 |
| GS9 + Male + 30s + 198 + 193 + 2011.3.26 + 2011.10.4 |
| GS10 + Male + 20s + 87 + 58 + 2011.9.30 + 2011.11.16 |
| GS12 + Male + 20s + 376 + 366 + 2010.11.14 + 2011.11.14 |
--------------------------------------------------------


##########################
Description of data values
##########################
The timestamp is stored as yyyyMMddHHmmssEEE
e.g., 20110509030752TUE means 2011-05-09 03:07:52 Tuesday

# table name
## column name
### comments

# locationTable
## _latitude, _longitude, _latitude_gps, _longitude_gps, _latitude_wifi, _longitude_wifi
### latitude and longitude in Earth with 10^(-6) precision.
### e.g., data value (123456789) means latitude or longtitude (123.456789) as degree value.
### _gps indicates data from GPS, _wifi indicates data from WPS, _latitude/_longitude is better one among gps or wps data.
### 0 = UNKNOWN_LOCATION
## _accuracy, _accuracy_gps, _accuracy_wifi
### error bound of location provided by Android in meters.
### 1 = user manually changed location information
### 100000 = UNKNOWN_ACCURACY
## _activity
### 1 = STATIONAY
### 2 = MOVE
## _place_name
### user-labeld place name. The data value is anonymized with one alphabet (e.g., A, B, C, and so on) and number (e.g., from 001 to 999).
## _time_location
### latest saved date as timestamp

# apTable
## _bssid
### anonymized mac address. 
### The uniqueness is maintained (i.e., same mac addresses indicates same APs in entire dataset).
## _signal
### average value of recived signal strength
## _signal_deviation
### deviation of recived signal strengths
## _time_ap
### latest saved date as timestamp

# cellTable
## _cell_type
### data from android.telephony.TelephonyManger.getPhoneType() 
### see http://developer.android.com/reference/android/telephony/TelephonyManager.html#getPhoneType()
### 0 = PHONE_TYPE_NONE
### 1 = PHONE_TYPE_GSM
### 2 = PHONE_TYPE_CDMA
### 3 = PHONE_TYPE_SIP
## _connet_time
### connection duration as milliseconds.
### the data is under-estimated since the system collected it in short burst.
## _time_cell
### latest saved date as timestamp

# stayTable
## _stay_time
### stay duration as milliseconds
## _stay_start_time
### start time (arrival time) of stay behavior as timestamp
## _time_stay
### end time (departure time) of stay behavior as timestamp

# batteryTable
## _battery_level
### Extra for ACTION_BATTERY_CHANGED: integer field containing the current battery level, from 0 to EXTRA_SCALE.
### see http://developer.android.com/reference/android/os/BatteryManager.html#EXTRA_LEVEL
## _battery_status
### Extra for ACTION_BATTERY_CHANGED: integer containing the current status constant.
### see http://developer.android.com/reference/android/os/BatteryManager.html#EXTRA_STATUS
### 1=BATTERY_STATUS_UNKNOWN, 2=BATTERY_STATUS_CHARGING, 3=BATTERY_STATUS_DISCHARGING, 4=BATTERY_STATUS_NOT_CHARGING, 5=BATTERY_STATUS_FULL
## _battery_voltage
### Extra for ACTION_BATTERY_CHANGED: integer containing the current battery voltage level.
### see http://developer.android.com/reference/android/os/BatteryManager.html#EXTRA_VOLTAGE
## _time_battery
### saved date as timestamp