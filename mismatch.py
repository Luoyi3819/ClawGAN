import random
import math
import cv2

def mismatch(img1_path, img2_path):
  
  img1 = cv2.imread(img1_path)
  img2 = cv2.imread(img2_path)
  
	H, W = img1.shape[:2]
	y = random.randint(0, W/2)   #random choose (x,y) from left top region 
	x = random.randint(0, H/2)
	template = img1[x:H//2 , y:W//2]   #cut the region as template
  
	result = cv2.matchTemplate(img2,template,cv2.TM_CCORR_NORMED)   #*******method********
	cv2.normalize( result, result, 0, 1, cv2.NORM_MINMAX, -1 )
  
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)  #get the max value from R
	dis = math.sqrt(pow(max_loc[0] - y , 2) + pow(max_loc[1] - x ,2)) 
	print("distance",dis)
	mm = dis/(abs(dis) + 1)
	return mm
