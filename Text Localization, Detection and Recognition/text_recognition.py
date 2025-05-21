# Tesseract OCR Text Recognition

# import the necessary packages
import numpy as np
import time
import cv2
from imutils.object_detection import non_max_suppression
import pytesseract


img = "E:\\Machine Learning\\Applied Data Science & AI\\Day8\\EAST Text Detection and Recognition with Tesseract OCR\\opencv-text-detection\\images\\car_wash.png"
east = "E:\\Machine Learning\\Applied Data Science & AI\\Day8\\EAST Text Detection and Recognition with Tesseract OCR\\opencv-text-detection\\frozen_east_text_detection.pb"
width = 320
height = 320
padding = 0.03
min_confidence = 0.5


# load the input image and grab the image dimensions
image = cv2.imread(img)
imageH, imageW = image.shape[:2]

#cv2.imshow("Input Image", image)
#cv2.waitKey(0)

# set the new width and height and then determine the ratio in change for both of them
rH = imageH / float(height)
rW = imageW / float(width)

# resize the image and grab the new image dimensions
image_resized = cv2.resize(image, (width, height))

#cv2.imshow("Resized Image", image_resized)
#cv2.waitKey(0)

# load the pre-trained EAST text detector NN model
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet(east)
net.getLayerNames()

# define the two output layer names for the EAST detector model that we are interested
# The first layer is our output sigmoid activation which gives us the probability of
# a region containing text or not.
# The second layer is the output feature map that represents the âgeometryâ of the image
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
	scoresData = scores[0, 0, y]
	xData0 = geometry[0, 0, y]
	xData1 = geometry[0, 1, y]
	xData2 = geometry[0, 2, y]
	xData3 = geometry[0, 3, y]
	anglesData = geometry[0, 4, y]

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

# initialize the list of results
results = []

# loop over the bounding boxes
for (startX, startY, endX, endY) in boxes:
	# scale the bounding box coordinates based on the respective ratios
    # draw the bounding box on the original image before resizing
	startX = int(startX * rW)
	startY = int(startY * rH)
	endX = int(endX * rW)
	endY = int(endY * rH)

    ## ENHANCEMENT
	# in order to obtain a better OCR of the text we can potentially apply a bit of padding
    # surrounding the bounding box -- here we compute the deltas in both x & y directions
	dX = int((endX - startX) * padding)
	dY = int((endY - startY) * padding)

	# apply padding to each side of the bounding box, respectively
	startX = max(0, startX - dX)
	startY = max(0, startY - dY)
	endX = min(imageW, endX + (dX * 2))
	endY = min(imageH, endY + (dY * 2))

	# extract the actual padded ROI "Region of Interest"
	roi = image[startY:endY, startX:endX]

	# in order to apply Tesseract v4 to OCR text we must supply (1) a language,
	# (2) an OEM "OCR Engine Mode", indicating algorithm we wish to use "1 for LSTM NN model",
	# (3) an PSM "Page Segmentation Mode", "7 implies that we treat ROI as a single line of text"
    
    # tesseract --help-oem: show available OCR Engine Modes:
    # 0 Legacy engine only, 1 Neural nets LSTM engine only, 2 Legacy + LSTM engines, 
    # 3 Default based on what is available.
    
    # tesseract --help-psm: show Page Segmentation Modes:
    # 0  Orientation and script detection (OSD) only.
    # 1  Automatic page segmentation with OSD.
    # 2  Automatic page segmentation, but no OSD, or OCR. (not implemented)
    # 3  Fully automatic page segmentation, but no OSD. (Default)
    # 4  Assume a single column of text of variable sizes.
    # 5  Assume a single uniform block of vertically aligned text.
    # 6  Assume a single uniform block of text.
    # 7  Treat the image as a single text line.
    # 8  Treat the image as a single word.
    # 9  Treat the image as a single word in a circle.
    # 10 Treat the image as a single character.
    # 11 Sparse text. Find as much text as possible in no particular order.
    # 12 Sparse text with OSD.
    # 13 Raw line. Treat the image as a single text line, bypassing hacks that are Tesseract-specific.
    
	config = ("-l eng --oem 1 --psm 7")
	text = pytesseract.image_to_string(roi, config=config)  #Localize and OCR text

	# add the bounding box coordinates and OCR'd text to the list of results
	results.append(((startX, startY, endX, endY), text))

# sort the results bounding box coordinates from top to bottom
results = sorted(results, key=lambda r:r[0][1])

# loop over the results
for ((startX, startY, endX, endY), text) in results:
	# display the text OCR'd by Tesseract
	print("OCR TEXT")
	print("========")
	print("{}\n".format(text))

	output = image.copy()
	cv2.rectangle(output, (startX, startY), (endX, endY), (0, 0, 255), 2)

	# show the output image
	cv2.imshow("Text Detection", output)
	cv2.waitKey(0)
