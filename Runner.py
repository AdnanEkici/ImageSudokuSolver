from PIL import Image
import cv2
import numpy as np
import Utils as u
import warnings
import CNN
import copy
import math
warnings.filterwarnings('ignore')





def distance_between(p1, p2):
    a = p2[0] - p1[0]
    b = p2[1] - p1[1]
    return np.sqrt((a ** 2) + (b ** 2))

def perpectiveTransform(img, corners):
    top_left, top_right, bottom_right, bottom_left = corners[0], corners[1], corners[2], corners[3]
    src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')
    side = max([
        distance_between(bottom_right, top_right),
        distance_between(top_left, bottom_left),
        distance_between(bottom_right, bottom_left),
        distance_between(top_left, top_right)
    ])
    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, m, (int(side), int(side)))

def infer_grid(img ,markCellCorners):
    cells = []
    side = img.shape[:1]
    side = side[0] / 9
    # i ile j nin yerini değiştirirsen yukarıdan aşağı doğru yazar.
    for j in range(9):
        for i in range(9):
            p1 = (i * side, j * side)
            p2 = ((i + 1) * side, (j + 1) * side)
            cells.append((p1, p2))

    # Kutuları doğru bulup bulmadığını kontrol et.
    if markCellCorners == 1:
        cornerDebug = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        for i in range(81):
            cornerDebug = cv2.circle(cornerDebug, (int(cells[i][0][0]) , int(cells[i][0][1])), radius=1, color=(255, 0, 255), thickness=8)#Magenta
            cornerDebug = cv2.circle(cornerDebug, (int(cells[i][0][0]), int(cells[i][1][0])), radius=1,color=(255, 0, 255), thickness=8)
            cornerDebug = cv2.circle(cornerDebug, (int(cells[i][0][1]), int(cells[i][0][1])), radius=1,color=(255, 0, 255), thickness=8)
            cornerDebug = cv2.circle(cornerDebug, (int(cells[i][0][1]), int(cells[i][0][0])), radius=1,color=(255, 0, 255), thickness=8)
            cornerDebug = cv2.circle(cornerDebug, (int(cells[i][0][1]), int(cells[i][1][0])), radius=1,color=(255, 0, 255), thickness=8)
            cornerDebug = cv2.circle(cornerDebug, (int(cells[i][1][1]), int(cells[i][0][0])), radius=1,color=(255, 0, 255), thickness=8)
            cornerDebug = cv2.circle(cornerDebug, (int(cells[i][1][1]), int(cells[i][1][0])), radius=1,color=(255, 0, 255), thickness=8)
        #cv2.imshow("Marked Cells", cornerDebug)
        #cv2.waitKey(0)
        cv2.imwrite("7.jpeg" , cornerDebug)

    else:
        print("Skipped Displaying: Mark Cells")
    return cells

def extract_digit(img, cell):
    digit = img[int(cell[0][1]):int(cell[1][1]), int(cell[0][0]):int(cell[1][0])]
    return digit

def get_digits(img, squares,filterChoose , displayOptions):
    digits = []
    img = u.pre_process_image(img.copy(),filterChoose , 4, skip_dilate=True)
    for square in squares:
        digits.append(extract_digit(img, square))
    return digits

def parse_grid(path , filterChoose , displayOptions , debugMode):
    original = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    processed = u.pre_process_image(original , filterChoose , displayOptions)
    #u.show_Image("Processed" , processed , debugMode)
    corners = u.get_field_corners(processed ,1)
    transformed = perpectiveTransform(original, corners)
    #u.show_Image("Perspective Transform", transformed, debugMode)
    cv2.imwrite("6.jpeg" , transformed)
    squares = infer_grid(transformed , 1)
    digits = get_digits(transformed, squares , filterChoose ,displayOptions)
    return digits


def denoise_image_with_connected_components(noisyDigitArray):
    denoised_digits = noisyDigitArray.copy()
    for i in range(len(denoised_digits)):
        denoised_digits[i] = u.denois_digit(noisyDigitArray[i])


    return denoised_digits


def clearify_image(image):

    rows,cols = image.shape
    for i in range(rows):
        for j in range(15):
            image[i][j] = 0

    for i in range(rows-1 , 0 , -1):
        for j in range(cols-1 , 50 , -1):
            image[i][j] = 0

    for i in range(5):
        for j in range(cols):
            image[i][j] = 0

    for j in range(cols-1 , 0 , -1):
        for i in range(rows-1 , 43 , -1):
            image[i][j] = 0

    return image

def get_white_black_ratio(digit):
    number_of_white_pix = np.sum(digit == 255)
    number_of_black_pix = np.sum(digit == 0)  # extracting only black pixels
    if math.isinf(number_of_black_pix/number_of_white_pix):
        return 999
    else:
        return  number_of_black_pix/number_of_white_pix

def isDigit(black_white_ratio):
    if 12 < black_white_ratio and 32 > black_white_ratio:
        return True
    else:
        return False

def clearify_image(image):

    rows,cols = image.shape
    for i in range(rows):
        for j in range(20):
            image[i][j] = 0

    for i in range(rows-1 , 0 , -1):
        for j in range(cols-1 , 50 , -1):
            image[i][j] = 0

    for i in range(10):
        for j in range(cols):
            image[i][j] = 0

    for j in range(cols-1 , 0 , -1):
        for i in range(rows-1 , 44 , -1):
            image[i][j] = 0

    return image


def find_threshholds_of_digits_debug(denoisedDigits):
        pad = 15
        for i in range(len(denoisedDigits)):
            resizedDigit = cv2.resize(clearify_image(denoisedDigits[i]), (28, 28))
            resizedDigit = cv2.copyMakeBorder(resizedDigit, pad, pad, pad, pad, cv2.BORDER_CONSTANT, (0, 0, 0))
            resizedDigit = cv2.resize(clearify_image(denoisedDigits[i]), (24, 20))
            cv2.imwrite("digit.png", resizedDigit)
            digit = cv2.imread("digit.png", 1)
            print(get_white_black_ratio(digit))
            u.show_Image("Digit", clearify_image(denoisedDigits[i]), 1)


if __name__ == '__main__':

   image_filename = "input5.jpeg"
   image = cv2.imread(image_filename , 0)

   noisySudokuDigits = parse_grid(image_filename , filterChoose=0 , displayOptions=0 , debugMode=0)
   deepCopyNoisyDigits = copy.deepcopy(noisySudokuDigits)
   denoisedDigits = denoise_image_with_connected_components(noisySudokuDigits)
   deepCopydenoisedDigits = copy.deepcopy(denoisedDigits)


   # for i in range(len(denoisedDigits)):
   #    cv2.imwrite("digit.png" , clearify_image(denoisedDigits[i]))
   #    digit = cv2.imread("digit.png" ,0)
   #    CNN.findDigit("digit.png")
   #    u.show_Image("Clearify", clearify_image(denoisedDigits[i]), 1)

   for i in range(len(denoisedDigits)):
       u.show_three_Image("Noisy --- Denoised --- Post Processed", deepCopyNoisyDigits[i], denoisedDigits[i], clearify_image(deepCopydenoisedDigits[i]) ,  2)

   sudoku = np.zeros((81))

   for i in range(len(denoisedDigits)):
       resizedDigit = cv2.resize(clearify_image(denoisedDigits[i]), (24, 20))
       u.show_Image("Resized: ", resizedDigit, 0)
       cv2.imwrite("digit.png", resizedDigit)
       digit = cv2.imread("digit.png", 1)
       if isDigit(get_white_black_ratio(clearify_image(denoisedDigits[i]))):
        sudoku[i] = CNN.findDigit("digit.png")
        u.show_Image("Digit", clearify_image(denoisedDigits[i]), 0)
       else:
        print("Empty")
        sudoku[i] = -1
        # u.show_Image("Empty", clearify_image(denoisedDigits[i]), 1)


   print(sudoku)



   # pad = 15
   # for i in range(len(denoisedDigits)):
   #     resizedDigit = cv2.resize(clearify_image(denoisedDigits[i]), (24, 20))
   #     cv2.imwrite("digit.png", resizedDigit)
   #     digit = cv2.imread("digit.png", 1)
   #     print(get_white_black_ratio(clearify_image(denoisedDigits[i])      ))
   #     u.show_Image("Digit", clearify_image(denoisedDigits[i]), 1)


