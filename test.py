import os
import math
import glob
import dlib
import cv2
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mping
from matplotlib import cm
from moviepy.editor import VideoFileClip
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from scipy.ndimage.measurements import label
import pickle

test_images = np.array([plt.imread(i) for i in glob.glob('../test_images/*.jpg')])
car_images = []
non_car_images = []

for root, dirs, files in os.walk('../datasets/vehicles/'):
    for file in files:
        if file.endswith('.png'):
            car_images.append(os.path.join(root, file))

for root, dirs, files in os.walk('../datasets/non-vehicles/'):
    for file in files:
        if file.endswith('.png'):
            non_car_images.append(os.path.join(root, file))

# Compute binned color features by scaling images down
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features


# Compute color histogram features
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


# Return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                  visualize=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                       visualize=vis, feature_vector=feature_vec)
        return features


# Extract feature wrapper that extracts and combines all features
def extract_features(imgs, cspace='RGB', orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0, spatial_size=(32, 32),
                     hist_bins=32, hist_range=(0, 256)):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mping.imread(file)
        # apply color conversion if other than 'RGB'

        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)

        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)

        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # Apply color_hist() also with a color space option now
        hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
        # Append the new feature vector to the features list
        features.append(np.concatenate((spatial_features, hist_features, hog_features)))

    # Return list of feature vectors
    return features


car_test = mping.imread(car_images[35])
car_test = cv2.cvtColor(car_test, cv2.COLOR_RGB2YCrCb)
non_car_test = mping.imread(non_car_images[20])
non_car_test = cv2.cvtColor(non_car_test, cv2.COLOR_RGB2YCrCb)

imgs = []
titles = []
for i in range(3):
    for feature_image, img_type in zip([car_test, non_car_test], ['Car', 'Non-car']):
        channel = feature_image[:, :, i]
        imgs.append(channel)
        titles.append(img_type + ' CH%d' % i)
        features, hog_image = get_hog_features(channel, orient=12, pix_per_cell=8, cell_per_block=2,
                                               vis=True, feature_vec=False)
        imgs.append(hog_image)
        titles.append(img_type + ' CH%d' % i + ' HOG')

fig, axes = plt.subplots(3, 4, sharex=True, sharey=True, figsize=(14, 10))
axes = axes.ravel()
for ax, img, title in zip(axes, imgs, titles):
    ax.imshow(img, cmap='Greys_r')
    ax.set_title(title)
    ax.axis('off')
plt.savefig('output_images/HOG_comparison.png')

print('Loading Classifier parameters...')
dist_pickle = pickle.load( open("svc_pickle.p", "rb" ) )
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial"]
hist_bins = dist_pickle["hist_bins"]

print('Loading is done!')

# Extracts features using hog sub-sampling and make predictions
def find_cars(img, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, ystart_ystop_scale,
              h_shift=0, visualisation=False):
    # List of bounding box positions
    bbox_detection_list = []
    box_vis_list = []
    # Copy and normalise
    draw_img = np.copy(img)
    img = img.astype(np.float32) / 255

    xstart = int(img.shape[0] /  5)
    xstop = img.shape[0]

    # Searching different size windows at different scales:
    for (ystart, ystop, scale) in ystart_ystop_scale:
        box_vis = []
        # Crop
        img_tosearch = img[ystart:ystop, :480, :]

        ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] /scale), np.int(imshape[0] / scale)))

        ch1 = ctrans_tosearch[:, :, 0]
        ch2 = ctrans_tosearch[:, :, 1]
        ch3 = ctrans_tosearch[:, :, 2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
        nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
        nfeat_per_block = orient * cell_per_block ** 2

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        # Compute individual channel HOG features for the entire image
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos * pix_per_cell
                ytop = ypos * pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

                # Get color features
                spatial_features = bin_spatial(subimg, size=spatial_size)
                hist_features = color_hist(subimg, nbins=hist_bins)

                # Scale features and make a prediction
                test_features = X_scaler.transform(
                    np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))

                # Make prediction based on trained model
                test_prediction = svc.predict(test_features)

                if (visualisation):
                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    win_draw = np.int(window * scale)
                    # Append Detection Position to list
                    box_vis.append(
                        ((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart)))

                if test_prediction == 1:
                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    win_draw = np.int(window * scale)
                    # Append Detection Position to list
                    bbox_detection_list.append(((xbox_left + h_shift, ytop_draw + ystart),
                                                (xbox_left + win_draw + h_shift, ytop_draw + win_draw + ystart)))
                    # Draw Detection on image
                    # cv2.rectangle(draw_img, (xbox_left + h_shift, ytop_draw + ystart),
                    #               (xbox_left + win_draw + h_shift, ytop_draw + win_draw + ystart), (0, 0, 255), 6)
        box_vis_list += [box_vis]
    return bbox_detection_list, draw_img, box_vis_list

test_image_sliding=test_images[0]

# Accumulation of labels from last N frames
class Detect_history():
    def __init__(self):
        # Number labels to store
        self.queue_len = 7  # 17 13
        self.queue = []

    # Put new frame
    def put_labels(self, labels):
        if (len(self.queue) > self.queue_len):
            tmp = self.queue.pop(0)
        self.queue.append(labels)

    # Get last N frames hot boxes
    def get_labels(self):
        detections = []
        for label in self.queue:
            detections.extend(label)
        return detections

# Read in image similar to one shown above
image = test_images[0]
blank = np.zeros_like(image[:,:,0]).astype(np.float)

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        print(bbox[0], bbox[1])
        cv2.rectangle(img, (bbox[0][0] - 150, bbox[0][1] - 50), (bbox[1][0] - 150, bbox[1][1] - 50), (0, 0, 255), 4)
        if (bbox[0][0] >= 200 and  bbox[0][0] <= 392 ) and (bbox[0][1] >= 130 and bbox[0][1] <= 260) \
            and (bbox[1][0] >= 400 and  bbox[1][0] <= 650) and (bbox[1][1] >= 450 and bbox[1][1] <= 500):

                cv2.rectangle(img, (bbox[0][0]-150,bbox[0][1]-50), (bbox[1][0]-150,bbox[1][1]-50), (0,0,255), 4)
    # Return the image
    return img



### Parameters
spatial = 32
hist_bins = 32
colorspace = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb #YCrCb best
orient = 9
pix_per_cell = 8
cell_per_block = 2
spatial_size = (32, 32)
heat_threshold = 4  # 12
hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL" #ALL,0 best

# ystart_ystop_scale = [(120, 150, 1.5), (150, 250, 3), (100, 480, 3.5)]
# ystart_ystop_scale = [(120, 150, 3), (180, 250, 2.5), (180, 480, 3)]

# best
# ystart_ystop_scale = [(100, 150, 1), (190, 250, 2.5), (160, 480, 3)]
# ystart_ystop_scale = [(100, 150, 1.5), (150, 250, 2), (170, 350, 3), (200, 480, 3.5)]

# 旁邊車道也會偵測到 ，615-S8影片有錯
# ystart_ystop_scale = [(100, 180, 1.5), (200, 250, 2.5), (160, 480, 3.5)]
# ystart_ystop_scale = [(100, 160, 1.5), (180, 250, 2.5), (130, 480, 3.5)]
ystart_ystop_scale = [(100, 150, 1), (120, 200, 1.5), (180, 250, 2), (200, 480, 2.5)]


def process_image(img):
    # Using Subsampled HOG windows to get possible detections
    bbox_detection_list, detections, box_vis_list = find_cars(img, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                                                              spatial_size, hist_bins, ystart_ystop_scale)

    flag = bbox_detection_list
    blank = np.zeros_like(img[:, :, 0]).astype(np.float)

    # Smoothing out previous detections
    detect_history.put_labels(bbox_detection_list)
    bbox_detection_list = detect_history.get_labels()

    # Add heat to detections
    heatmap = add_heat(blank, bbox_detection_list)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)

    # Draw bounding box
    result = draw_labeled_bboxes(np.copy(img), labels)

    return result ,flag

detect_history = Detect_history()

# negative = cv2.imread('../test_images/0.jpg')
# positive = cv2.imread('../test_images/3.jpg')
# result1 = positive[:,200:500, :]
# result, flag = process_image(result1)
# plt.figure(figsize = (20,20))
# plt.imshow(result)
# plt.show()
# plt.axis("off")


# current camera
cap = cv2.VideoCapture('../videos/615-S8_0-210825-171010-171510-20010100.mp4')
dlib.image_window()

while (cap.isOpened()):
   ret, frame = cap.read()
   result, flag = process_image(frame)
   cv2.line(result, (200, 300), (500, 300), (255, 0, 0), 3)
   cv2.imshow('frame', result)
   # if len(flag) > 0:
   #      print('with cars!')
   # else:
   #     print('no cars!')
   key = cv2.waitKey(1)
   # ESC
   if key == 27:
      breakcap.release()
cv2.destroyAllWindows()






