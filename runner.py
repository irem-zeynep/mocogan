import os
import time
import argparse

logFile = open("executionTimes.csv", "a")

parser = argparse.ArgumentParser(description='Start trainning MoCoGAN.....')
parser.add_argument('--pre-train', type=int, default=0,
                     help='set 1 if you want to use pretrained models')

args       = parser.parse_args()
pre_train  = args.pre_train

while True:
	start_time = time.time()
	if pre_train:
		numberFile = open("lastTrainedNumber.txt", "r")
		iterationNumber = numberFile.readline()
		os.system(("python3 train.py --niter 10000 --pre-train {}").format(iterationNumber))
		numberFile.close()
	else:
		os.system("python3 train.py --niter 10000")
	logFile.write("Start time: {}, lastTrainNumber: {}\n".format(start_time, iterationNumber))
	logFile.flush()
	
logFile.close()
