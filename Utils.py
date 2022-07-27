import cv2 as cv
import numpy as np
import operator



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

def show_two_Image(title,image1 , image2 ,displayImages):
    if(displayImages == 0):
        print("Skipped Displaying:" , title)
    elif(displayImages == 1):
        numpy_horizontal_concat = np.concatenate((image1, image2), axis=1)
        cv.imshow(title, numpy_horizontal_concat)
        cv.waitKey(0)
    elif(displayImages == 2):
        numpy_horizontal_concat = np.concatenate((image1, image2), axis=1)
        cv.imshow(title, numpy_horizontal_concat)
        cv.waitKey(0)
        cv.destroyAllWindows()


def show_three_Image(title,image1 , image2, image3 ,displayImages):
    if(displayImages == 0):
        print("Skipped Displaying:" , title)
    elif(displayImages == 1):
        numpy_horizontal_concat = np.concatenate((image1, image2 , image3), axis=1)
        cv.imshow(title, numpy_horizontal_concat)
        cv.waitKey(0)
    elif(displayImages == 2):
        image1 = cv.resize(image1, (60, 60))
        image2 = cv.resize(image2, (60, 60))
        image3 = cv.resize(image3, (60, 60))
        numpy_horizontal_concat = np.concatenate((image1, image2 , image3), axis=1)
        cv.imshow(title, numpy_horizontal_concat)
        cv.waitKey(0)
        cv.destroyAllWindows()

def pre_process_image(image , filterType , displayOption ,skip_dilate=False):
    show_Image("Input Image" , image , displayOption)

    if filterType == 0:
        proc = cv.GaussianBlur(image.copy(), (7, 7), 0)
        show_Image("Gaussian Blurred", proc, displayOption)
        if displayOption == 4:
            cv.imwrite("8.jpeg" , proc)
        else:
            cv.imwrite("1.jpeg" , proc)
    elif filterType == 1:
        print("Box Filter")
    else:
        print("Median Filter")



    proc = cv.adaptiveThreshold(proc, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)# Adaptive threshold using 11 nearest neighbour pixels
    show_Image("Adaptive Threshhold", proc, displayOption)
    if displayOption == 4:
        cv.imwrite("9.jpeg" , proc)
    else:
        cv.imwrite("2.jpeg", proc)

    proc = cv.bitwise_not(proc, proc)
    show_Image("Invertion", proc, displayOption)
    if displayOption == 4:
        cv.imwrite("10.jpeg" , proc)
    else:
        cv.imwrite("3.jpeg", proc)

    if not skip_dilate:
        kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
        proc = cv.dilate(proc, kernel)
        show_Image("Dilated Image", proc, displayOption)
        if displayOption == 4:
            cv.imwrite("11.jpeg", proc)
        else:
            cv.imwrite("4.jpeg", proc)

    return proc

def get_field_corners(img ,mark_corners):
    contours, h = cv.findContours(img.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # Find contours
    contours = sorted(contours, key=cv.contourArea, reverse=True)  # Sort by area, descending
    polygon = contours[0]  # Largest image

    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))

    # Köşeleri doğru bulup bulmadığını kontrol et.
    if mark_corners == 1 or mark_corners == 4:
        cornerDebug = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        cornerDebug = cv.circle(cornerDebug, (polygon[top_left][0]), radius=1, color=(0, 255, 0), thickness=8);print("Sol Üst Köşe: " , polygon[top_left][0])#Yeşil
        cornerDebug = cv.circle(cornerDebug, (polygon[top_right][0]), radius=1, color=(0, 0, 255), thickness=8);print("Sağ Üst Köşe: " , polygon[top_right][0])#Kırmızı
        cornerDebug = cv.circle(cornerDebug, (polygon[bottom_right][0]), radius=1, color=(255, 0, 0), thickness=8);print("Sağ Alt Köşe: " , polygon[bottom_right][0])#Mavi
        cornerDebug = cv.circle(cornerDebug, (polygon[bottom_left][0]), radius=1, color=(255, 255, 0), thickness=8);print("Sol Alt Köşe: " ,polygon[bottom_left][0])#Cyan
        #show_Image("Corners Of Game Field" , cornerDebug , 2)
        if mark_corners == 4:
            cv.imwrite("", cornerDebug)
        else:
            cv.imwrite("5.jpeg", cornerDebug)
    else:
        print("Skipped Displaying:  Marked Corners")
    return [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]

def denois_digit(image):
    img = image
    labelnum, labelimg, contours, _ = cv.connectedComponentsWithStats(img)
    for label in range(1, labelnum):
        x, y, w, h, size = contours[label]
        if size <= 50:
            img[y:y + h, x:x + w] = 0
    return img


