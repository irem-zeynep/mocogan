import os
import time

logFile = open("executionTimes.csv", "a")

while True:
	start_time = time.time()
	numberFile = open("lastTrainedNumber.txt", "r")
	iterationNumber = numberFile.readline()
	os.system(("python3 train.py --niter 10000 --pre-train {}").format(iterationNumber))	
	logFile.write("Start time: {}, lastTrainNumber: {}\n".format(start_time, iterationNumber))
	logFile.flush()
	numberFile.close()
	
logFile.close()
