import numpy as np
import cv2


filename = 'input2.jpeg'
img = cv2.imread(filename)
mapper = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 2, 3, 0.24)
# result is dilated for marking the corners, not important
dst = cv2.dilate(dst, None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst > 0.01 * dst.max()] = [0, 255, 0]

rows, cols, _ = img.shape

points = []

for i in range(rows):
    for j in range(cols):
        if (img[i][j][1] == 255):
            points.append((i, j))

for i in range(len(points)):
    print(points[i])

pts1 = np.float32([[0, 260], [640, 260],
                   [0, 400], [640, 400]])
pts2 = np.float32([[0, 0], [400, 0],
                   [0, 640], [400, 640]])
insidePoints={}
size=len(points)



# # Apply Perspective Transform Algorithm
# matrix = cv2.getPerspectiveTransform(pts1, pts2)
# print(matrix)
# result = cv2.warpPerspective(mapper, matrix, (500, 600))
# Sol Üst Köşe:  [  5 125]
# Sağ Üst Köşe:  [527 113]
# Sağ Alt Köşe:  [551 459]
# Sol Alt Köşe:  [ 18 478]
# cv2.imshow('Trasnformed', result)  # Transformed Capture
#æ


cv2.imshow('dst',img)
cv2.waitKey(0)