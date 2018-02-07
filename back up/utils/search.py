
# import the necessary packages
from colordescriptor import ColorDescriptor
from searcher import Searcher
import argparse
import cv2
from imutils import build_montages
import datetime
import math 

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--index", required = True,
	help = "Path to where the computed index will be stored")
ap.add_argument("-q", "--query", required = True,
	help = "Path to the query image")
ap.add_argument("-r", "--result-path", required = True,
	help = "Path to the result path")
args = vars(ap.parse_args())
 
# initialize the image descriptor
cd = ColorDescriptor((8, 12, 3))

# load the query image and describe it
query = cv2.imread(args["query"])
features = cd.describe(query)
 
# perform the search
searcher = Searcher(args["index"])
results = searcher.search(features,limit = 50)
 
# display the query
cv2.imshow("Query", query)
images = []
# # loop over the results
for (score, resultID) in results:
 	# load the result image and display it
 	result = cv2.imread(args["result_path"] + "/" + resultID)
 	cv2.putText(result, str(score)[:3], (65, 219), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 0), 1)
 	images.append(result)

montages = build_montages(images, (128, 196), (8, 8))
# loop over the montages and display each of them
for montage in montages:
	cv2.imshow("Montage", montage)
	timestamp = datetime.datetime.now()
	start_time = timestamp.strftime("%Y%m%d-%H%M%S")
	outputPath = "{}/{}.jpg".format(args["result_path"] , start_time)
	cv2.imwrite(outputPath, montage)
	cv2.waitKey(0)