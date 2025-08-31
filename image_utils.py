# --- align_imgs function for image alignment ---
import cv2
import numpy as np

def align_imgs(img1, img2, max_features=500, good_match_percent=0.15):
	"""
	Align img2 to img1 using feature matching and homography.
	Returns the aligned version of img2 and the homography matrix.
	"""
	# Convert images to grayscale
	im1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	im2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

	# Detect ORB features and compute descriptors.
	orb = cv2.ORB_create(max_features)
	keypoints1, descriptors1 = orb.detectAndCompute(im1_gray, None)
	keypoints2, descriptors2 = orb.detectAndCompute(im2_gray, None)

	# Match features.
	matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
	matches = matcher.match(descriptors1, descriptors2, None)

	# Sort matches by score
	matches = sorted(matches, key=lambda x: x.distance)

	# Remove not so good matches
	num_good_matches = int(len(matches) * good_match_percent)
	matches = matches[:num_good_matches]

	# Extract location of good matches
	points1 = np.zeros((len(matches), 2), dtype=np.float32)
	points2 = np.zeros((len(matches), 2), dtype=np.float32)

	for i, match in enumerate(matches):
		points1[i, :] = keypoints1[match.queryIdx].pt
		points2[i, :] = keypoints2[match.trainIdx].pt

	# Find homography
	h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

	# Use homography
	height, width, channels = img1.shape
	img2_aligned = cv2.warpPerspective(img2, h, (width, height))

	return img2_aligned, h
"""
Adapted image utilities for document verification.
"""
# All necessary functions should be defined locally in this file. If you need any specific function from the old counterfeit.utils.image_utils, copy its code here.
