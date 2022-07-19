from PIL import Image
import cv2
import numpy as np
import Utils as u
import warnings
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
        cv2.imshow("Marked Cells", cornerDebug)
        cv2.waitKey(0)

    else:
        print("Skipped Displaying: Mark Cells")
    return cells

def extract_digit(img, cell):
    digit = img[int(cell[0][1]):int(cell[1][1]), int(cell[0][0]):int(cell[1][0])]
    return digit

def get_digits(img, squares,filterChoose , displayOptions):
    digits = []
    img = u.pre_process_image(img.copy(),filterChoose , displayOptions, skip_dilate=True)
    for square in squares:
        digits.append(extract_digit(img, square))
    return digits

def parse_grid(path , filterChoose , displayOptions , debugMode):
    original = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    processed = u.pre_process_image(original , filterChoose , displayOptions)
    u.show_Image("Processed" , processed , debugMode)
    corners = u.get_field_corners(processed ,debugMode)
    transformed = perpectiveTransform(original, corners)
    u.show_Image("Cropped", transformed, debugMode)
    squares = infer_grid(transformed , debugMode)
    digits = get_digits(transformed, squares , filterChoose ,displayOptions)
    return digits



if __name__ == '__main__':

   image = cv2.imread("input5.jpeg" , 0)
   sudokuDigits = parse_grid("input5.jpeg" , filterChoose=0 , displayOptions=2 , debugMode=1)
   for i in range(81):
        noised_image = sudokuDigits[i].copy()
        u.show_two_Image("Noised - Denoised" , noised_image, u.clear_image(sudokuDigits[i]) , 2)