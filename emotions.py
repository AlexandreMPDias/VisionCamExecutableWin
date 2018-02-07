# USAGE
# python emotion_detector.py --cascade haarcascade_frontalface_default.xml --model checkpoints/epoch_75.hdf5

# import necessary for pyinstaller
from scipy import optimize

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils import build_montages
import numpy as np
import argparse
import imutils
from imutils.video import VideoStream
import cv2
import httplib2
import time
from imutils import paths
import random
import datetime
import operator
import tenor

import random 

def translate_emotion(emotion):
	return {
		"angry": 'nervoso',
		"scared": 'assustado',
		"happy": 'alegre',
		"sad": 'triste',
		"surprised": 'surpreso',
		"neutral": 'neutro'
	}[emotion]

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade",  default="haarcascade_frontalface_default.xml",
	help="path to where the face cascade resides")
ap.add_argument("-m", "--model", default="checkpoints/epoch_75.hdf5",
	help="path to pre-trained emotion detector CNN")
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-a", "--alpha", type=float, default=0.5,
	help="alpha transparency of the overlay (smaller is more transparent)")
ap.add_argument("-f", "--face", 
	help="if face True show face")

ap.add_argument("-r", "--resolution", default="1920x1080",
	help="resolution of the screen defautl 1920x1080 ")
args = vars(ap.parse_args())

PNG_PATH = "./png"
EMOTIONS_PATH = "./gifs"


ten = tenor.Tenor()

# load the face detector cascade, emotion detection CNN, then define
# the list of emotion labels
detector = cv2.CascadeClassifier(args["cascade"])
model = load_model(args["model"])
EMOTIONS = ["angry", "scared", "happy", "sad", "surprised","neutral"]
CONTROLS_EMOTIONS_TRESH  ={
	"emotions": {
		"angry" : 90,
		"happy" : 80,
		"neutral" : 80 ,
		"sad" : 51,
		"scared" :51,
		"surprised" : 52,

	}
}
CONTROLS_GIFS_FEEDBACK  ={
	"emotions": {
		"angry" : ["nervoso","irritado","to de mal" , "chateado","angry", "bad","furious","ta nervoso","are you angry"],
		"happy" : ["sorriso","felicidade","Alegria","enjoy","alegre" , "feliz","sorriso","smile to the world", "happy day","so happy","i love you smile","dog smile","this is good","cute smile"],
		"neutral" : ["Oi","ta tudo bem" ,"como voce vai" ,"Hi you","hello", "indifferent","dont care","so what","thinking","dont give a fuck"],
		"sad" : ["sad", "unhappy","so unhappy","sorry"],
		"scared" : ["scared", "omg","so scared","boo","freak"],
		"surprised" : ["surprised", "surpresa","omg","surprise","so surprised","super surprised","wtf"],

	}
}



def saveVideo(emotion,timestamp, frames):
	outputPath = "{}/{}/{}.mp4".format(EMOTIONS_PATH,emotion, timestamp.strftime("%Y%m%d-%H%M%S"))
	h, w, _ = frames[0].shape

	# fourcc = cv2.CV_FOURCC(*'XVID')
	fourcc = cv2.VideoWriter_fourcc(*"MJPG")
	out = cv2.VideoWriter(outputPath, fourcc, 20.0, (w,h))

	for frame in frames:
		out.write(frame)

	out.release()
	return outputPath

def grab_frames(url):
	frames = []
	video =  cv2.VideoCapture(url)
	while True:
		grab , frame = video.read()
		if grab == True:
			frames.append(frame)
		else:
			return frames


# def emotionVideo(emotion):
# 	gif_url = ten.randommp4(emotion)
# 	print(gif_url)
# 	frames = grab_frames(gif_url)
# 	timestamp = datetime.datetime.now()
# 	saveVideo(emotion,timestamp, frames)
# 	for frame in frames :
# 		frame = imutils.resize(frame, width=880)
# 		cv2.imshow("frames",frame)
# 		time.sleep(0.1)
# 	# if the 'q' key is pressed, stop the loop
# 		if cv2.waitKey(1) & 0xFF == ord("q"):
# 			break
# 	cv2.destroyWindow("frames")

def watermark(image,mark_path,position="center",alpha=args["alpha"]):
	watermark = mark_path
	(wH, wW) = watermark.shape[:2]
	(B, G, R, A) = cv2.split(watermark)
	B = cv2.bitwise_and(B, B, mask=A)
	G = cv2.bitwise_and(G, G, mask=A)
	R = cv2.bitwise_and(R, R, mask=A)
	watermark = cv2.merge([B, G, R, A])
	(h, w) = image.shape[:2]
	image = np.dstack([image, np.ones((h, w), dtype="uint8") * 255])
 
	overlay = np.zeros((h, w, 4), dtype="uint8")

	if position =="right_low" :
		overlay[h - wH - 10:h - 10, w - wW - 10:w - 10] = watermark
	if position == "center" :
		overlay[int(h/2) - int(wH/2) :int(h/2)+int(wH/2) , int(w/2) - int(wW/2) :int(w/2)+int(wW/2)] = watermark
	if position == "center_right" :
		overlay[int(h/2) - int(wH/2) :int(h/2)+int(wH/2) , int(w/2) :int(w/2)+int(wW)] = watermark
	if position == "center_left" :
		overlay[int(h/2) - int(wH/2) :int(h/2)+int(wH/2) , int(w/2)-int(wW) :int(w/2)] = watermark

 	# blend the two images together using transparent overlays
	output = image.copy()
	cv2.addWeighted(overlay, alpha, output, 1.0, 0, output)
	return output

def insert_animation(image,frame,position="center",alpha=args["alpha"]):
	watermark = frame

	(wH, wW) = watermark.shape[:2]
	(h, w) = image.shape[:2]
	overlay =  np.zeros((h, w, 3), dtype="uint8")
	print("watermark :"+str(watermark.shape[:2]))
	print("frame window  shape" + str(image.shape[:2]))
	if position =="right_low" :
		overlay[h - wH - 10:h - 10, w - wW - 10:w - 10] = watermark
	if position == "center" :
		init_y = int(h/2) - int(wH/2)
		if init_y < 0:
			init_y = 0 
		print("init_y :"+str(init_y))
	
		overlay[init_y :init_y+int(wH) , int(w/2) - int(wW/2) :int(w/2)+int(wW/2)] = watermark
	if position == "center_left" :
		overlay[int(h/2) - int(wH/2) :int(h/2)+int(wH/2) , int(w/2)-int(wW) :int(w/2)] = watermark

 
	# blend the two images together using transparent overlays
	output = image.copy()
	cv2.addWeighted(overlay, alpha, output, 1.0, 0, output)
 
	# write the output image to disk
	return output

def draw_stats(frame,enum):
	canvas = np.zeros((220, 600, 3), dtype="uint8")
	(h, w) = frame.shape[:2]
	(wH,wW) = canvas.shape[:2]

	x_start= int(wW/2)
	y_start= 0

	print(x_start,y_start)
	for (i, (emotion, prob)) in enum:
		
		color = (0, 255,0)
		if prob *100 < 51 :
			fullfill = -1
		else:
			fullfill = -1
		# construct the label text
		text = "{}".format(emotion)
		wg = int(prob * 300)
		cv2.rectangle(canvas, (x_start + 0, ((i+y_start) * 35) + 5 ),
			(x_start + wg, ((i+y_start) * 35) + 35 ), color, fullfill)
		cv2.putText(canvas, translate_emotion(text), (10, ((i+y_start) * 35) + 33 ),
			cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2) 
		frame[int(h/2) - int(wH/2) :int(h/2)+int(wH/2) , int(w/2)-int(wW) :int(w/2)] =canvas
	return frame

def draw_face(frame,face_frame):
	(h, w) = frame.shape[:2]
	(wH,wW) = face_frame.shape[:2]
	frame[h - wH - 10:h - 10, w - wW - 10:w - 10] = face_frame
	return frame


def show_animation(emotion):
	print(emotion)
	frame_montage = montage_copy.copy()
	subject = random.choice(CONTROLS_GIFS_FEEDBACK["emotions"][emotion])
	cv2.putText(frame_montage, subject, (10, frame_montage.shape[0]-10),
				cv2.FONT_HERSHEY_SIMPLEX, 2.45,
				(255, 255, 255), 2)
	gif_url = ten.randommp4(subject)
	print(gif_url)
	frames = grab_frames(gif_url)
	timestamp = datetime.datetime.now()
	
	for framex in frames :
		frame_to_show = imutils.resize(framex, width=880, height=880)
		(h, w) = frame_to_show.shape[:2]
		h_montage = frame_montage.shape[0]
		if h > h_montage:
			print("crop image at :" + str(h_montage))
			frame_to_show=frame_to_show[0:h_montage,0:w]
		x_montage =  insert_animation(frame_montage,frame_to_show,alpha=1.0)
		cv2.imshow("Montage",x_montage)
		
		time.sleep(0.01)
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break
	saveVideo(emotion,timestamp, frames)

	start_time = time.time()

def show_local_animation(emotion):
	print(emotion)
	frame_montage = montage_copy.copy()
	print(EMOTIONS_PATH+"/"+emotion)
	imagePaths = list(paths.list_files(EMOTIONS_PATH+"/"+emotion,validExts=(".mp4")))
	print(imagePaths)
	random.shuffle(imagePaths)
	
	if imagePaths == []: print("need more gifs at :" + emoticon)
	imagePath = imagePaths[0]
	subject = translate_emotion(emotion)

	cv2.putText(frame_montage, subject, (10, frame_montage.shape[0]-10),
				cv2.FONT_HERSHEY_SIMPLEX, 2.45,
				(255, 255, 255), 2)
	frames = grab_frames(imagePath)
	
	for framex in frames :
		frame_to_show = imutils.resize(framex, width=880, height=880)
		(h, w) = frame_to_show.shape[:2]
		h_montage = frame_montage.shape[0]
		if h > h_montage:
			print("crop image at :" + str(h_montage))
			frame_to_show=frame_to_show[0:h_montage,0:w]
		x_montage =  insert_animation(frame_montage,frame_to_show,alpha=1.0)
		cv2.imshow("Montage",x_montage)
		
		# time.sleep(0.01)
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break
	# saveVideo(emotion,timestamp, frames)

	
### Main ###############

# if a video path was not supplied, grab the reference to the webcam
if not args.get("video", False):
	camera = VideoStream(src=0, usePiCamera=False).start()

# otherwise, load the video
else:
	camera = VideoStream(src=args["video"], usePiCamera=False).start()	

# keep looping
cv2.namedWindow('Montage', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Montage', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

start_time = time.time()

# Draw main window 
WIN_max_x = round(int(args["resolution"].split('x')[0])/1)
WIN_max_y = round(int(args["resolution"].split('x')[1])/1)
#montage = build_montages([], (256, 256), (WIN_max_x, WIN_max_y))[0]
montage = np.zeros((WIN_max_y, WIN_max_x, 3), dtype="uint8")
montage_copy = montage.copy()
emotionPath = "{}/{}.png".format(PNG_PATH, "sunglass")
alpha = args["alpha"]
preds =[0.0, 0.0, 0.0, 0.0, 0.0,0.0]
enum = enumerate(zip(EMOTIONS, preds))
tresh=False
last_emotion = None

while True:
	frame = camera.read()
	#for test olnly 
	if frame is None:
		camera.stop()
		print("frame none")
				# if a video path was not supplied, grab the reference to the webcam
		if not args.get("video", False):
			camera = VideoStream(src=0, usePiCamera=False).start()

		# otherwise, load the video
		else:
			camera = VideoStream(src=args["video"], usePiCamera=False).start()	
		frame = camera.read()

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rects = detector.detectMultiScale(gray, scaleFactor=1.3, 
		minNeighbors=5, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)

	#if no faces detected show smile and stats labels only
	if len(rects) == 0 :

		enum = enumerate(zip(EMOTIONS, preds))
		draw_stats(montage,enum)
		frame_montage =  watermark(montage,cv2.imread(emotionPath, cv2.IMREAD_UNCHANGED),position="center_right" , alpha=1.0)
		cv2.imshow("Montage",frame_montage)
		
	# ensure at least one face was found before continuing
	elif len(rects) > 0:
		# montage = montage_copy.copy()
		# determine the largest face area
		rect = sorted(rects, reverse=True,
			key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
		(fX, fY, fW, fH) = rect
       
		# extract the face ROI from the image, then pre-process
		# it for the network
		
		roi = gray[fY:fY + fH, fX:fX + fW]
		roi = cv2.resize(roi, (48, 48))
		roi = roi.astype("float") / 255.0
		roi = img_to_array(roi)
		roi = np.expand_dims(roi, axis=0)
		xfY = fY - 50
		xfH = fH + 150 
		xfX = fX - 50 
		xfW = fW + 150 
		if xfY + xfH > frame.shape[0] : 
			Ybotton = frame.shape[0]-10
		else :
			Ybotton =xfY + xfH 
		if xfX + xfW > frame.shape[1] :
			Xmax = frame.shape[1] -10 
		else :
			Xmax = xfX + xfW
		if xfY < 0 : 
			xfY = 0
		if xfX < 0 :
			xfX = 0 

		if  args.get("face", True):
			draw_face(montage,frame[xfY:Ybotton, xfX:Xmax])
		# make a prediction on the ROI, then lookup the class
		# label
		preds = model.predict(roi)[0]
		label = EMOTIONS[preds.argmax()]
		elapse = (time.time() - start_time)
		
		print("Make a emotion")
		
		enum = enumerate(zip(EMOTIONS, preds))

		
		for (i, (emotion, prob)) in enum:
			print(prob*100)
			if prob*100 > 51:
		 		emotionPath = "{}/{}.png".format(PNG_PATH, emotion)
		 		if prob*100 >CONTROLS_EMOTIONS_TRESH["emotions"][emotion]:
		 			last_emotion = emotion
		 			tresh = True
		enum = enumerate(zip(EMOTIONS, preds))

		draw_stats(montage,enum)
		emoticon = cv2.imread(emotionPath, cv2.IMREAD_UNCHANGED)

		frame_montage =  watermark(montage,emoticon,position="center_right" , alpha=1.0)
		cv2.imshow("Montage",frame_montage)

		if tresh == True:
			time.sleep(1)
			if cv2.waitKey(1) & 0xFF == ord("q"):
				break	
			show_local_animation(last_emotion)	
			tresh = False


	if cv2.waitKey(1) & 0xFF == ord("q"):
		break	

# cleanup the camera and close any open windows

cv2.destroyAllWindows()
