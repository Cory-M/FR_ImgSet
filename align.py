import cv2
import vipl
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import os

from matlab_cp2tform import get_similarity_transform_for_cv2

fd_path = "models/VIPLFaceDetector5.1.0.sta"
fec_pd_path = "models/SeetaPointDetector2.0.pts5.ats"
pd_path = "models/VIPLPointDetector5.0.pts5.dat"

face = 0
no_face = 0

def alignment(src_img,src_pts):
    ref_pts = [ [30.2946, 51.6963],[65.5318, 51.5014],
        [48.0252, 71.7366],[33.5493, 92.3655],[62.7299, 92.2041] ]
    crop_size = (96, 112)
    src_pts = np.array(src_pts).reshape(5,2)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    return face_img

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
					_pts = []
					for i in range(5):
						_pts.append([pts[i].x,pts[i].y])

					face_img = alignment(src_img=img,src_pts=_pts)
					cv2.imwrite(dst_path,face_img)
					face =face + 1
			
				else :
					no_face = no_face + 1
	print(face)
	print(no_face)



