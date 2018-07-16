import cv2
import vipl
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import os

fd_path = "models/VIPLFaceDetector5.1.0.sta"
fec_pd_path = "models/SeetaPointDetector2.0.pts5.ats"
pd_path = "models/VIPLPointDetector5.0.pts5.dat"

face = 0
no_face = 0

if __name__ == "__main__":

	# initialize VIPL detectors
	detector = vipl.Detector(fd_path)
	predictor = vipl.Predictor(pd_path)
	cropper = vipl.ExtCrop()

	img_src = '/media/zhineng/Data/M/aligned_images_DB'
	img_dst = '/media/zhineng/Data/M/aligned_imgs'
	for speaker in os.listdir(img_src):
		if os.path.exists(img_dst + '/' + speaker):
			continue
		
		os.mkdir(img_dst + '/' +speaker)
		spk_path = img_src +'/' + speaker
		for video in os.listdir(spk_path):
			if not os.path.exists(img_dst + '/' + speaker + '/' + video):
				os.mkdir(img_dst + '/' + speaker + '/' + video)
			video_path = spk_path +'/' + video
			print(video_path)
			for imgs in os.listdir(video_path):
				path = video_path +'/' +imgs
				dst_path = img_dst + '/' + speaker + '/' + video + '/' + imgs
				
				img = cv2.imread(path)		
				faces = detector(img)
				#detector.draw_boxes(img) # preview face detection result in boxes

				if len(faces) > 0 :
					bbox = (faces[0].x, faces[0].y, faces[0].w, faces[0].h)		
					pts = predictor(img, bbox[0], bbox[1], bbox[2], bbox[3])
					size = 256
					rate = 1.0
					crop_pts = cropper.crop(img , pts , size , rate)
					crop_img = cropper.cropped_image
					crop_img = cv2.cvtColor(crop_img,cv2.COLOR_BGR2RGB)
					misc.imsave(dst_path, crop_img)
					face =face + 1
			
				else :
					no_face = no_face + 1		
	print(face)
	print(no_face)



