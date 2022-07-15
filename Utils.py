import cv2 as cv
import numpy as np


#Displays Image
def show_Image(title,image ,displayImages):
    if(displayImages == 0):
        print("Skipped Displaying: " , title)
    elif(displayImages == 1):
        cv.imshow(title, image)
        cv.waitKey(0)
    elif(displayImages == 2):
        cv.imshow(title, image)
        cv.waitKey(0)
        cv.destroyAllWindows()


#Prints matris to the console for debugging
def print_matris(matris):
    print(matris[0])
    for i in range(matris[0]):
        print()
        for j in range(matris[1]):
            print(matris[i][j] + " "),




def denoise_random_white_points(image):
    cnts = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv.contourArea(c)
        if area < 1000:
            cv.drawContours(image, [c], -1, (0, 0, 0), -1)
    return image

def perspective_transform(image, corners):
    def order_corner_points(corners):
        # Separate corners into individual points
        # Index 0 - top-right
        #       1 - top-left
        #       2 - bottom-left
        #       3 - bottom-right
        corners = [(corner[0][0], corner[0][1]) for corner in corners]
        top_r, top_l, bottom_l, bottom_r = corners[0], corners[1], corners[2], corners[3]
        return (top_l, top_r, bottom_r, bottom_l)

    # Order points in clockwise order
    ordered_corners = order_corner_points(corners)
    top_l, top_r, bottom_r, bottom_l = ordered_corners

    # Determine width of new image which is the max distance between
    # (bottom right and bottom left) or (top right and top left) x-coordinates
    width_A = np.sqrt(((bottom_r[0] - bottom_l[0]) ** 2) + ((bottom_r[1] - bottom_l[1]) ** 2))
    width_B = np.sqrt(((top_r[0] - top_l[0]) ** 2) + ((top_r[1] - top_l[1]) ** 2))
    width = max(int(width_A), int(width_B))

    # Determine height of new image which is the max distance between
    # (top right and bottom right) or (top left and bottom left) y-coordinates
    height_A = np.sqrt(((top_r[0] - bottom_r[0]) ** 2) + ((top_r[1] - bottom_r[1]) ** 2))
    height_B = np.sqrt(((top_l[0] - bottom_l[0]) ** 2) + ((top_l[1] - bottom_l[1]) ** 2))
    height = max(int(height_A), int(height_B))

    # Construct new points to obtain top-down view of image in
    # top_r, top_l, bottom_l, bottom_r order
    dimensions = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1],
                    [0, height - 1]], dtype = "float32")

    # Convert to Numpy format
    ordered_corners = np.array(ordered_corners, dtype="float32")

    # Find perspective transform matrix
    matrix = cv.getPerspectiveTransform(ordered_corners, dimensions)

    # Return the transformed image
    return cv.warpPerspective(image, matrix, (width, height))


def create_clean_board(image , displayOptions , filterChoose):


    # Parameters for selection
    chooseFilter = displayOptions  # 0 = gaussianBlur , 1 = boxBlur , 2 = medianBlur
    displayImages = displayOptions  # 0 = Do not show images , 1 = show images stacks , 2 = destroy image

    # Read Image
    image = cv.imread(image, 0)  # Read image as greyscale
    show_Image("Input Image", image, displayImages)

    # Denoise Image
    denoisedImage = image
    if (chooseFilter == 0):
        denoisedImage = cv.GaussianBlur(image, (13, 13), 1.4)
    elif (chooseFilter == 1):
        print("Box Blur")
    else:
        print("Median Blur")

    show_Image("Deinoised Image", denoisedImage, displayImages)

    # TrashHolding
    tresholdedImage = cv.adaptiveThreshold(denoisedImage, 255,
                                           cv.ADAPTIVE_THRESH_MEAN_C | cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,
                                           45, 5)
    show_Image("Threshold Applied", tresholdedImage, displayImages)

    invertedImage = cv.bitwise_not(tresholdedImage)
    show_Image("Invertion", invertedImage, displayImages)

    # Dilate the image
    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
    # dilatedImage = cv.dilate(invertedImage, kernel)
    dilatedImage = invertedImage
    # show_Image("Dilated Image", dilatedImage , displayImages)

    # binaryCleanImage = denoise_random_white_points(dilatedImage)
    binaryCleanImage = dilatedImage
    show_Image("Denoised Image", binaryCleanImage, displayImages)

    cnts = cv.findContours(binaryCleanImage, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv.contourArea, reverse=True)
    for c in cnts:
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.015 * peri, True)
        transformed = perspective_transform(binaryCleanImage, approx)
        break

    transformed = cv.erode(transformed, kernel)
    show_Image("Trasnformed Image", transformed, displayImages)
    cv.imwrite("board.png" , transformed)













