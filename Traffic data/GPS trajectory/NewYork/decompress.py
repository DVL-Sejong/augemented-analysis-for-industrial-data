# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 16:57:42 2014

@author: Brian Donovan (briandonovan100@gmail.com)

This script decompresses the entire dataset.  It reads a set of files like FOIL2010/trip_data_1.zip
and creates a new set of files like FOIL2010/trip_data_1.csv.  Parallel processing can be used if it is available.
The directory structure is copied.  Non-existing files are ignored.
You may want to change NUM_PROCESSORS, COMPRESSED_FOLDER_NAME, or DECOMPRESSED_FOLDER_NAME depending on your machine.
This script should work on a Linux/Mac machine which has python and unzip installed.  It will NOT work on Windows,
but it should be fairly straightforward to change the commands to use winzip.
"""

import csv
import shutil
import os
from multiprocessing import Pool
from datetime import datetime


program_start = datetime.now()
#A convenient print statement for long runs - also includes a timestamp at the beginning of the message
#Arguments:
	#msg - a string to be printed
def logMsg(msg):
	td = datetime.now() - program_start
	print "[" + str(td) + "]  " + str(msg)




#Change this to a number <= the number of available CPUs on your computer
NUM_PROCESSORS = 1

#Change these to reflect the folder names on your machine
COMPRESSED_FOLDER_NAME = "compressed"
DECOMPRESSED_FOLDER_NAME = "decompressed"




#Decompresses the trip and fare files for a month
def processMonth((year, month)):
	
	try:
		#Generate filenames
		target_folder = os.path.join(DECOMPRESSED_FOLDER_NAME, "FOIL" + str(year))
		tripFile = os.path.join(COMPRESSED_FOLDER_NAME, "FOIL" + str(year), "trip_data_" + str(month) + ".zip")
		fareFile = os.path.join(COMPRESSED_FOLDER_NAME, "FOIL" + str(year), "trip_fare_" + str(month) + ".zip")
		
		
		#Decompress trip file
		cmd = "unzip " + tripFile + " -d " + target_folder
		logMsg(cmd)
		os.system(cmd)
		
		#Decompress fare file
		cmd = "unzip " + fareFile + " -d " + target_folder
		logMsg(cmd)
		os.system(cmd)
	except:
		logMsg("FILE NOT FOUND : " + tripFile + "  . Skipping...")


#An iterator which goes through all months of the 4 year dataset
def monthIterator():
	for y in range(2010, 2014):
		for m in range(1, 13):
			yield (y, m)








logMsg("Setting up folders...")
#Remove old folder if it exists
shutil.rmtree(DECOMPRESSED_FOLDER_NAME, ignore_errors=True)

#Create folder hierarchy
os.mkdir(DECOMPRESSED_FOLDER_NAME)
os.mkdir(os.path.join(DECOMPRESSED_FOLDER_NAME, "FOIL2010"))
os.mkdir(os.path.join(DECOMPRESSED_FOLDER_NAME, "FOIL2011"))
os.mkdir(os.path.join(DECOMPRESSED_FOLDER_NAME, "FOIL2012"))
os.mkdir(os.path.join(DECOMPRESSED_FOLDER_NAME, "FOIL2013"))


logMsg("Processing...")
#Process months in parallel
pool = Pool(NUM_PROCESSORS)
pool.map(processMonth, monthIterator())

logMsg("Done.")
