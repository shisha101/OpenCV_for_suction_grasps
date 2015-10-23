import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt
import time
import os.path
from copy import deepcopy


def drawMatches(img1, kp1, img2, kp2, matches):
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated 
    keypoints, as well as a list of DMatch data structure (matches) 
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint 
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)


    # Show the image
#     cv2.imshow('Matched Features', out)
#     cv2.waitKey(0)
#     cv2.destroyWindow('Matched Features')

    # Also return the image if you'd like a copy
    return out

# algorithm parameters
number_of_kps = 10000 # the number of top kps to be used for comparison
number_of_kps_homagraphy = 10 # the minumum number of kps used to find the homgrapy between train and image
number_of_kps_used_for_hoography = 50
max_n_kp = 500 # default value max number of key points

# input and output parameters
visulaize = False
save_matching_images = True
save_images_with_kp = True
number_of_kps_to_plot = 5000
start_image_num = 0
end_image_num = 30

# path parameters
query_path = "Query_images/"
template_path =""
output_path = "matched_images/"
template_name = "18_blured_resize.jpg"

#declarations
query_image_names =[]
query_images = []
distance_measurment = []
kp_query_list = []
des_query_list = []
matches_list = []
homography_list = []

#start time of complete algorithm
time_start = time.time()

#create list of image names
for index in xrange(start_image_num, end_image_num):
    string_temp = query_path+str(index)+".jpg"
    if(os.path.exists(string_temp)):
        query_image_names.append(string_temp)
    else:
        print "the file with the following path does not exist"
        print string_temp
if len(query_image_names) == 0:
    print "******** ERROR no query frames ********"
    sys.exit()


# read all images 
for index, image in enumerate(query_image_names):
    query_images.append(cv2.imread(image,0))
#     query_images[index] = cv2.GaussianBlur(query_images[index], (5, 5), 0)
if(os.path.exists(template_path+template_name)):
    img_template = cv2.imread(template_path+template_name, 0)
else:
    print "******** ERROR no template frames ********"
    sys.exit()
#     img_template = cv2.GaussianBlur(img_template, (5, 5), 0)


time_processing_start = time.time()
orb = cv2.ORB(max_n_kp*4, nlevels=8, WTA_K=2, patchSize=31, scaleFactor=1.2)#, WTA_K=4, patchSize=31) # http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_feature2d/py_orb/py_orb.html
orb2 = cv2.ORB(max_n_kp)
# kp extraction template image
# kp_template = orb.detect(img_template,None)
kp_template, des_temp = orb.detectAndCompute(img_template,None)

# kp extraction query image
for index, image in enumerate(query_images):
#     kp_query_list.append(orb.detect(image, None))
    kp_query, des_query = orb2.detectAndCompute(image, None)
    des_query_list.append(des_query)
    kp_query_list.append(kp_query)

# feature matcher
kp_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) # the flag enables cross checking this probably increases the run time significantly by a factor of 2
for index, descriptors in enumerate(des_query_list):
    matches = kp_matcher.match(des_temp, descriptors)
    matches = sorted(matches, key = lambda x:x.distance) #sort
    matches_list.append(matches)#store
#     print "the lenght of the matches is"
#     print len(matches)
    average_distance = 0
    if len(matches)< number_of_kps:
        number_of_kps = len(matches)
    if len(matches) == 0:
        average_distance = -99999
        print "no matches were found"
        continue
    for x in xrange(0,number_of_kps):
        average_distance += matches[x].distance
    average_distance = average_distance/number_of_kps
    distance_measurment.append(average_distance)
    if len(matches)>= number_of_kps_homagraphy:
        kp_homog = matches[:number_of_kps_used_for_hoography]
        src_pts = np.float32([kp_template[m.queryIdx].pt for m in kp_homog ]).reshape(-1,1,2)
        dst_pts = np.float32([kp_query_list[index][m.trainIdx].pt for m in kp_homog ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h,w = img_template.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        cv2.line(query_images[index], (int(dst[0][0][0]),int(dst[0][0][1])), (int(dst[1][0][0]),int(dst[1][0][1])), (255, 0, 0), 3)
        cv2.line(query_images[index], (int(dst[1][0][0]),int(dst[1][0][1])), (int(dst[2][0][0]),int(dst[2][0][1])), (255, 0, 0), 3)
        cv2.line(query_images[index], (int(dst[2][0][0]),int(dst[2][0][1])), (int(dst[3][0][0]),int(dst[3][0][1])), (255, 0, 0), 3)
        cv2.line(query_images[index], (int(dst[0][0][0]),int(dst[0][0][1])), (int(dst[3][0][0]),int(dst[3][0][1])), (255, 0, 0), 3)
#         img_temp = cv2.polylines(query_images[index],[np.int32(dst)],True,255,3, cv2.CV_AA)


print "total image processing time is: " + str (time.time()-time_processing_start)

with open("results.txt", "w") as f:
    line_in_1 = "average distance     query image     template image"  +"\n"
    for index, name in enumerate(query_image_names):
        
        input_line = format(distance_measurment[index], "03.4f")+ " ---> "+name.rsplit('/', 1)[-1] +" ---> "+ template_name +"\n"
        f.write(input_line)
distance_measurment_sorted = sorted(distance_measurment)
print distance_measurment_sorted
# matches = kp_matcher.knnMatch(des_temp, des_query, 2)

# visulaization and result saving
if visulaize or save_matching_images:
    for index in xrange(0,len(matches_list)):
        out = drawMatches(img_template, kp_template, query_images[index], kp_query_list[index], matches_list[index][:number_of_kps_to_plot])
        if visulaize:
            cv2.imshow('Matched Features '+str(index), out)
            cv2.waitKey(0)
            cv2.destroyWindow('Matched Features '+str(index))
        if save_matching_images:
            cv2.imwrite(output_path+'matching_output_'+str(index)+"_distance_"+str(distance_measurment[index])+'.jpg',out)
# Draw first 10 matches.


# cv2.drawMatchesKnn




if save_images_with_kp:
    img_template_with_kp = cv2.drawKeypoints(img_template, kp_template,color=(0 , 0, 255), flags=0) # ploting kp on image
    cv2.imwrite('img_template_with_kp.jpg',img_template_with_kp)
    print "the number of kps in the template image is : ", len(kp_template)
    img_query_with_kp = cv2.drawKeypoints(query_images[0], kp_query_list[0],color=(0 , 0, 255), flags=0) # ploting kp on image
    cv2.imwrite('img_query_with_kp.jpg',img_query_with_kp)
    print "the number of kps in the saved query image is : ", len(kp_query_list[0])

print "total elapsed time is: " +str(time.time()-time_start)

