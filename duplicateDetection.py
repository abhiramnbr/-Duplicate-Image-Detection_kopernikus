import cv2
import glob
import numpy as np
import PIL
from PIL import Image
import os
import glob
import imutils
from numpy.lib.type_check import imag

image_list1=[]		#for storing the first image
image_list2=[]		#for storing the second iamge
list_area=[]		#for storing the areas of contours
removeList=[]		#for storing the file names to be deleted


def main():
	#specify the image folder path
	path= "/Users/abhiram/Desktop/kopernickus/trial/"
	
	#counters
	iteration1=0
	iteration2=0
	imgLoop=0
	
	#loop1	
	for image1 in glob.glob(path+'*.png'):

		counter=0

		imgLoop+=1

		print(" ")

		print("Image {}".format(imgLoop))

		file1 = os.path.join(path, image1)

		image1=cv2.imread(file1)
		
		img1_func=preprocess_image_change_detection(image1)
		
		image_list1.append(img1_func)
		
		minContourArea1=imgAverage(img1_func)
		
		print(" ")
		
		print("Minimum Contour Area 1:",minContourArea1)

		curr=image_list1[iteration1]

		iteration1+=1

		#loop2
		for image2 in glob.glob(path+'*.png'): #change according to the file types		
			
			file2 = os.path.join(path, image2)
			
			image2=cv2.imread(file2)

			img2_func=preprocess_image_change_detection(image2)
			
			image_list2.append(img2_func)

			minContourArea2=imgAverage(img2_func)

			print("Minimum Contour Area 2:",minContourArea2)

			minContourArea=(minContourArea1+minContourArea2)/2

			prev=image_list2[iteration2]

			scr,cnts,img_thresh=compare_frames_change_detection(prev,curr,minContourArea)
			
			iteration2+=1

			if scr==0 and minContourArea2==minContourArea1:

				counter+=1
				
				if counter>1:
					
					print("Duplicate")
					
					if file2 not in removeList:
						
						removeList.append(file2)
						

	print("Deleting files:.{}".format(removeList))
	
	for i in removeList:
		
		os.remove(i)
	print(" ")
	print("Duplicates deleted!!")
	print(" ")
			

def imgAverage(img):

		#performing an adaptive threshold method 
		thresh1 = cv2.adaptiveThreshold(img.copy(), 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 21)

		# find contours in the thresholded image
		cnts = cv2.findContours(thresh1, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

		cnts = imutils.grab_contours(cnts)

		#looping over the detected contours
		for c in cnts:

			#computing each contour area
			arr=cv2.contourArea(c)

			if arr >5: #neglecting very small countours

				list_area.append(arr)

		#finding the average of the contour areas
		avg1=sum(list_area)/len(list_area)


		list_area.clear()

		return avg1
		

def draw_color_mask(img, borders, color=(0, 0, 0)):

	h = img.shape[0]

	w = img.shape[1]


	x_min = int(borders[0] * w / 100)

	x_max = w - int(borders[2] * w / 100)

	y_min = int(borders[1] * h / 100)

	y_max = h - int(borders[3] * h / 100)



	img = cv2.rectangle(img, (0, 0), (x_min, h), color, -1)

	img = cv2.rectangle(img, (0, 0), (w, y_min), color, -1)

	img = cv2.rectangle(img, (x_max, 0), (w, h), color, -1)

	img = cv2.rectangle(img, (0, y_max), (w, h), color, -1)


	return img


def preprocess_image_change_detection(img, gaussian_blur_radius_list=None, black_mask=(0, 37.5, 0, 0)):

	gray = img.copy()

	gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

	if gaussian_blur_radius_list is not None:

		for radius in gaussian_blur_radius_list:

			gray = cv2.GaussianBlur(gray, (radius, radius), 0)



	gray = draw_color_mask(gray, black_mask)



	return gray


def compare_frames_change_detection(prev_frame, next_frame, min_contour_area):

	frame_delta = cv2.absdiff(prev_frame, next_frame)
	
	thresh = cv2.threshold(frame_delta, 45, 255, cv2.THRESH_BINARY)[1]

	thresh = cv2.dilate(thresh, None, iterations=2)

	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,

							cv2.CHAIN_APPROX_SIMPLE)

	cnts = imutils.grab_contours(cnts)



	score = 0

	res_cnts = []

	for c in cnts:

		if cv2.contourArea(c) < min_contour_area:

			continue



		res_cnts.append(c)

		score += cv2.contourArea(c)



	return score, res_cnts, thresh


if __name__ == "__main__":
	main()