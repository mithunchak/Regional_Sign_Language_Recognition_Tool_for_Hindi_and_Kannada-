import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

# Initialize the video capture
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Parameters
offset = 20
imgsize = 300

folder = "data(new)/tha"
counter = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        # Convert cropped region to grayscale
        gray = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 2)

        # Create binary image using adaptive thresholding
        th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        ret, binary_image = cv2.threshold(th3, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Resize grayscale image to imgsize x imgsize
        aspectRatio = h / w
        if aspectRatio > 1:
            imgResized = cv2.resize(gray, (int(imgsize / aspectRatio), imgsize))
        else:
            imgResized = cv2.resize(gray, (imgsize, int(imgsize * aspectRatio)))

        # Create a white background image of size imgsize x imgsize
        imgWhite = np.ones((imgsize, imgsize), np.uint8) * 255

        # Place the resized grayscale image onto imgWhite
        h_resized, w_resized = imgResized.shape
        y_start = (imgsize - h_resized) // 2
        x_start = (imgsize - w_resized) // 2
        imgWhite[y_start:y_start + h_resized, x_start:x_start + w_resized] = imgResized

        # Create final grayscale image without draw
        img_final1 = np.ones((400, 400), np.uint8) * 148
        h1 = gray.shape[0]
        w1 = gray.shape[1]
        img_final1[((400 - h1) // 2):((400 - h1) // 2) + h1, ((400 - w1) // 2):((400 - w1) // 2) + w1] = gray

        # Create final binary image
        img_final = np.ones((400, 400), np.uint8) * 255
        h2 = binary_image.shape[0]
        w2 = binary_image.shape[1]
        img_final[((400 - h2) // 2):((400 - h2) // 2) + h2, ((400 - w2) // 2):((400 - w2) // 2) + w2] = binary_image

        # Skeleton drawing on a white background
        skeleton = np.ones((400, 400, 3), dtype=np.uint8) * 255  # White background
        pts = hand['lmList']  # List of 21 landmark points

        os = ((400 - w) // 2) - 15
        os1 = ((400 - h) // 2) - 15
        for t in range(0, 4, 1):
            cv2.line(skeleton, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1),
                     (0, 255, 0), 3)
        for t in range(5, 8, 1):
            cv2.line(skeleton, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1),
                     (0, 255, 0), 3)
        for t in range(9, 12, 1):
            cv2.line(skeleton, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1),
                     (0, 255, 0), 3)
        for t in range(13, 16, 1):
            cv2.line(skeleton, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1),
                     (0, 255, 0), 3)
        for t in range(17, 20, 1):
            cv2.line(skeleton, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1),
                     (0, 255, 0), 3)
        cv2.line(skeleton, (pts[5][0] + os, pts[5][1] + os1), (pts[9][0] + os, pts[9][1] + os1), (0, 255, 0),
                 3)
        cv2.line(skeleton, (pts[9][0] + os, pts[9][1] + os1), (pts[13][0] + os, pts[13][1] + os1), (0, 255, 0),
                 3)
        cv2.line(skeleton, (pts[13][0] + os, pts[13][1] + os1), (pts[17][0] + os, pts[17][1] + os1),
                 (0, 255, 0), 3)
        cv2.line(skeleton, (pts[0][0] + os, pts[0][1] + os1), (pts[5][0] + os, pts[5][1] + os1), (0, 255, 0),
                 3)
        cv2.line(skeleton, (pts[0][0] + os, pts[0][1] + os1), (pts[17][0] + os, pts[17][1] + os1), (0, 255, 0),
                 3)

        for i in range(21):
            cv2.circle(skeleton, (pts[i][0] + os, pts[i][1] + os1), 2, (0, 0, 255), 1)

        cv2.imshow("skeleton", skeleton)

        # Display images for debugging
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Skeleton_Image_{time.time()}.jpg',skeleton)
        print(counter)

    if key == ord("q"):  # Press 'q' to quit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
