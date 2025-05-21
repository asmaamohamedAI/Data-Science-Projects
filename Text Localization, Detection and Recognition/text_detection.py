# EAST Text Detection
# EAST : Efficient and Accurate Scene Text

# import the necessary packages
import numpy as np
import time
import cv2
from imutils.object_detection import non_max_suppression

#image path
img = "H:\\Markov Course cont\\Session 8\\Session8\\car_wash.png"
#Model path
east = "H:\\Markov Course cont\\Session 8\\Session8\\frozen_east_text_detection.pb"
#The EAST text requires that your input image dimensions be multiple of 32
#so if you choose to adjust your width and height values,make sure that they are multiple of 32
height = 320
width = 320
min_confidence = 0.5 #threshold


# load the input image and grab the image dimensions
image = cv2.imread(img)
imageH, imageW = image.shape[:2]

# show the image
cv2.imshow("Input Image", image)
cv2.waitKey(0)

# set the new width and height and then determine the ratio in change for both of them
rH = imageH / float(height)
rW = imageW / float(width)

# resize the image and grab the new image dimensions
image_resized = cv2.resize(image, (width, height))

cv2.imshow("Resized Image", image_resized)
cv2.waitKey(0)

# load the pre-trained EAST text detector NN model
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet(east)
net.getLayerNames()

# define the two output layer names for the EAST detector model that we are interested
# The first layer is our output sigmoid activation which gives us the probability of
# a region containing text or not.
# The second layer "concat3" is the output feature map that represents the “geometry” of the image
# we use this geometry to derive the bounding box coordinates of the text in the input image

layerNames = [net.getLayerNames()[-1],     #"feature_fusion/Conv_7/Sigmoid"
              net.getLayerNames()[-3]]     #"feature_fusion/concat_3"


# construct a blob from the image and then perform a forward pass of
# the model to obtain the two output layer sets

## [blobFromImage] creates 4-dimensional blob from image. Optionally resizes and crops image from center,
# subtract mean values, scales values by scalefactor, swap Blue and Red channels.
# blobFromImage: performs **Mean subtraction, **Scaling, And optionally **channel swapping

# blob output dimensions "NCHW = (batch_size, channel, height, width)"
# Mean subtraction is used to help combat illumination changes in the input images to aid our CNN
# mean values for the ImageNet training set are R=103.93, G=116.77, and B=123.68
# Scaling factor, sigma, adds in a normalization = 1.0
# Channel swapping: OpenCV assumes images are in BGR channel order; however, the `mean` value assumes
# we are using RGB order. To resolve this discrepancy we can swap the R and B channels in image.

blob = cv2.dnn.blobFromImage(image_resized,
                             scalefactor = 1.0,
                             size = (width, height),
                             mean=(123.68, 116.78, 103.94), #mean pixel intensity across all training images
                             swapRB=True, #swap first and last channels in 3-channel image is necessary
                             crop = False)

# pass the blob through EAST netwotk
# By supplying layerNames as a parameter to net.forward, we are instructing OpenCV to return the two feature maps:
# 1. The output scores map provides the probability of a given region containing text
# 2. The output geometry map used to derive the bounding box coordinates of text in our input image
start = time.time()
net.setInput(blob)
scores, geometry = net.forward(layerNames)
end = time.time()

# show timing information on text prediction
print("[INFO] text detection took {:.6f} seconds".format(end - start))


# grab the number of rows and columns from the scores volume, then initialize our set
# of bounding box rectangles and corresponding confidence scores
numRows, numCols = scores.shape[2:4]
rects = []
confidences = []

# loop over the number of rows
for y in range(0, numRows):
	# extract the scores (probabilities), followed by the geometrical data
    # used to derive potential bounding box coordinates that surround text
	scoresData = scores[0, 0, y, :]
	xData0 = geometry[0, 0, y, :]
	xData1 = geometry[0, 1, y, :]
	xData2 = geometry[0, 2, y, :]
	xData3 = geometry[0, 3, y, :]
	anglesData = geometry[0, 4, y, :]

	# loop over the number of columns
	for x in range(0, numCols):
		# if our score does not have sufficient probability, ignore it
		if scoresData[x] < min_confidence:
			continue

		# compute the offset factor as our resulting feature maps will
		# be 4x smaller than the input image ""input image = (320,320) & current image = (80,80)""
        # The EAST naturally reduces volume size as the image passes through the network
        # our volume size is actually 4x smaller than our input image.
        # so we multiply by four to bring the coordinates back into respect of our original image
		offsetX, offsetY = (x * 4.0, y * 4.0)

		# use the geometry volume to derive the width and height of the bounding box
        # h_upper, w_right, h_lower, w_left, A = geometry[0,:,y,x]
		h = xData0[x] + xData2[x]        # upper/lower
		w = xData1[x] + xData3[x]        # right/left

		# extract the rotation angle for the prediction and then compute the sin and cosine
		angle = anglesData[x]
		cos = np.cos(angle)
		sin = np.sin(angle)
        
		# compute both the starting and ending (x, y)-coordinates for
		# the text prediction bounding box
		endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
		endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
		startX = int(endX - w)
		startY = int(endY - h)

		# add the bounding box coordinates and probability score to our respective lists
		rects.append((startX, startY, endX, endY))
		confidences.append(scoresData[x])


# apply non-maxima suppression to suppress weak, overlapping bounding boxes
boxes = non_max_suppression(np.array(rects), probs=confidences)

# loop over the bounding boxes
for (startX, startY, endX, endY) in boxes:
	# scale the bounding box coordinates based on the respective ratios
    # draw the bounding box on the original image before resizing
	startX = int(startX * rW)
	startY = int(startY * rH)
	endX = int(endX * rW)
	endY = int(endY * rH)

	# draw the bounding box on the image
    # image, starting points, ending points, color, box thickness
	cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

# show the output image
cv2.imshow("Text Detection", image)
cv2.waitKey(0)