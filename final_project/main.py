import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image, ImageTk
from tkinter import messagebox
import threading
from keras.models import load_model
from keras.preprocessing import image

loaded_model = load_model("model.h5")

image_processing = None
max_image_size=(800,600)


def mosaic_image(image):
    windowSize = 5

    height, width = image.shape[:2]
    buffer = np.zeros(image.shape, np.uint8)

    # make mosaic image
    for i in range(0, height - windowSize, windowSize):
        for j in range(0, width - windowSize, windowSize):
            sumRed = sumGreen = sumBlue = 0

            for k in range(windowSize):
                for l in range(windowSize):
                    sumRed += image[i + k][j + l][0]
                    sumGreen += image[i + k][j + l][1]
                    sumBlue += image[i + k][j + l][2]

            sumRed = sumRed / (windowSize * windowSize)
            sumGreen = sumGreen / (windowSize * windowSize)
            sumBlue = sumBlue / (windowSize * windowSize)

            for k in range(windowSize):
                for l in range(windowSize):
                    buffer[i + k][j + l] = [sumRed, sumGreen, sumBlue]    

    cv2.imwrite("mosaic.jpg", buffer)


def convert_to_hsv(image):
    # HSV로 변환전 BGR채널을 나눌 공간.
    blue = np.zeros(image.shape[:2], np.uint8)
    green = np.zeros(image.shape[:2], np.uint8)
    red = np.zeros(image.shape[:2], np.uint8)

    # HSV 저장 공간.
    hsv_img = np.zeros(image.shape, np.uint8)

    # blue, green, red = cv2.split(image)
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            blue[row, col] = image[row, col, 0]
            green[row, col] = image[row, col, 1]
            red[row, col] = image[row, col, 2]

    # cv2.cvtColor(cv2.COLOR_BGR2HSV)
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            b = blue[row, col]
            g = green[row, col]
            r = red[row, col]

            v = max(r, g, b)
            s = ((v-min(r,g,b)) / v) * 255

            b_  = b/255.0; g_ = g/255.0; r_ = r/255.0

            num = ((r_-g_) + (r_-b_)) * 0.5
            den = np.sqrt((r_-g_)**2 + (r_-b_) * (g_-b_))

            if den: theta = math.acos(num/den) * (180/np.pi)
            else: theta = 0

            if b <= g: h = theta
            else:  h = 360-theta

            hsv_img[row, col, 0] = round(h/2)
            hsv_img[row, col, 1] = int(round(s))
            hsv_img[row, col, 2] = int(v)
            
    return hsv_img


def convert_to_rgb(image):
    height, width, channels = image.shape

    rgb_image = np.zeros_like(image, dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            h, s, v = image[i, j]

            # HSV를 RGB로 변환
            if s == 0:
                r = g = b = v
            else:
                h /= 60
                i = int(h)
                f = h - i
                p = v * (1 - s)
                q = v * (1 - s * f)
                t = v * (1 - s * (1 - f))

                if i == 0:
                    r, g, b = v, t, p
                elif i == 1:
                    r, g, b = q, v, p
                elif i == 2:
                    r, g, b = p, v, t
                elif i == 3:
                    r, g, b = p, q, v
                elif i == 4:
                    r, g, b = t, p, v
                else:
                    r, g, b = v, p, q

            rgb_image[i, j] = [b, g, r]

    return rgb_image

def inRange(image, lower_bound, upper_bound):
    # 마스크를 초기화합니다. 모든 값이 0인 배열을 생성합니다.
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # 이미지 내의 모든 픽셀을 순회합니다.
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # 현재 픽셀이 하한과 상한 사이에 있는지 확인합니다.
            if all(lower_bound <= image[i, j]) and all(image[i, j] <= upper_bound):
                mask[i, j] = 255

    return mask

def bitwise_and_with_mask(src, mask):
    # Check if the mask is valid
    if mask.shape != src.shape[:2] or mask.dtype != np.uint8:
        raise ValueError("Mask must be the same size as input array and of type uint8")

    # Initialize an output array with zeros, having the same shape as the input array
    output = np.zeros_like(src)

    # Apply the mask manually
    for i in range(src.shape[2]):  # Iterate over each color channel
        channel = src[:, :, i]
        masked_channel = np.bitwise_and(channel, mask)
        output[:, :, i] = masked_channel

    return output

def custom_bitwise_xor(src1, src2):
    # 입력 배열이 동일한 크기와 타입인지 확인합니다.
    if src1.shape != src2.shape or src1.dtype != src2.dtype:
        raise ValueError("source must be the same size as input array and of type uint8")

    # 비트 단위 XOR 연산을 수행합니다.
    return np.bitwise_xor(src1, src2)


def enhance_image(image):
    # 이미지를 HSV로 변환
    hsv_image = convert_to_hsv(image)

    # 잇몸 부분(빨간색) 감지
    lower_red = np.array([0, 80, 80])
    upper_red = np.array([10, 255, 255])
    red_mask = inRange(hsv_image, lower_red, upper_red)

    lower_pink = np.array([150, 80, 80])
    upper_pink = np.array([179, 255, 255])
    pink_mask = inRange(hsv_image, lower_pink, upper_pink)

    mask_gum = red_mask + pink_mask

    gum_part = bitwise_and_with_mask(image, mask_gum)
    gum_part = np.clip(gum_part + [20, 0, 30], 0, 255)  # 빨간색 강도 증가
    gum_part = bitwise_and_with_mask(gum_part, mask_gum)

    tooth_part = bitwise_and_with_mask(image, ~mask_gum)
    tooth_part = bitwise_and_with_mask(tooth_part, ~mask_gum)

    gum_part = gum_part.astype(np.uint8)
    tooth_part = tooth_part.astype(np.uint8)

    enhanced_image = cv2.bitwise_xor(gum_part, tooth_part)

    cv2.imwrite("enhanced_image.jpg", enhanced_image)


def delete_gum(image):
    hsv_image = convert_to_hsv(image)
    lower_red = np.array([0, 80, 80])
    upper_red = np.array([10, 255, 255])
    red_mask = inRange(hsv_image, lower_red, upper_red)

    lower_pink = np.array([150, 80, 80])
    upper_pink = np.array([179, 255, 255])
    pink_mask = inRange(hsv_image, lower_pink, upper_pink)

    mask_gum = red_mask + pink_mask

    # 잇몸 부분을 검은색으로 칠하기
    image_without_gum = bitwise_and_with_mask(image, ~mask_gum)

    # 결과 저장
    cv2.imwrite("image_without_gum.jpg", image_without_gum)


def blur_image(image):
    blurMask = np.array([[ 1,  2,  1],
                         [ 2,  4,  2],
                         [ 1,  2,  1]])
    blurWeight = 16

    height, width = image.shape[:2]
    buffer = np.zeros(image.shape, np.uint8)

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            sumRed = sumGreen = sumBlue = 0

            for k in range(3):
                for l in range(3):
                    sumRed += blurMask[k][l] * image[i + k - 1][j + l - 1][0]
                    sumGreen += blurMask[k][l] * image[i + k - 1][j + l - 1][1]
                    sumBlue += blurMask[k][l] * image[i + k - 1][j + l - 1][2]

            sumRed = min(max(sumRed / blurWeight, 0), 255)
            sumGreen = min(max(sumGreen / blurWeight, 0), 255)
            sumBlue = min(max(sumBlue / blurWeight, 0), 255)

            buffer[i][j] = [sumRed, sumGreen, sumBlue]

    cv2.imwrite("blurred.jpg", buffer)


def binarize_image(image):
    threshold = 120

    height, width = image.shape[:2]

    for i in range(height):
        for j in range(width):
            if image[i][j][0] > threshold and image[i][j][1] > threshold and image[i][j][2] > threshold:
                image[i][j] = [255, 255, 255]
            else:
                image[i][j] = [0, 0, 0]

    cv2.imwrite("binarized.jpg", image)



def apply_mask(image, mask, result, edgeThreshold):
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            sumRed = sumGreen = sumBlue = 0

            for k in range(3):
                for l in range(3):
                    sumRed += mask[k][l] * image[i + k - 1][j + l - 1][0]
                    sumGreen += mask[k][l] * image[i + k - 1][j + l - 1][1]
                    sumBlue += mask[k][l] * image[i + k - 1][j + l - 1][2]

            sumRed = min(max(sumRed, 0), 255)
            sumGreen = min(max(sumGreen, 0), 255)
            sumBlue = min(max(sumBlue, 0), 255)

            if sumRed >= edgeThreshold and sumGreen >= edgeThreshold and sumBlue >= edgeThreshold:
                result[i][j] = [255, 0, 0]



def edge_detection(image):
    Horizontal = np.zeros(image.shape, np.uint8)
    Vertical = np.zeros(image.shape, np.uint8)
    DiagonalRight = np.zeros(image.shape, np.uint8)
    DiagonalLeft = np.zeros(image.shape, np.uint8)
    final = np.zeros(image.shape, np.uint8)

    edgeThreshold = 120

    HorizontalMask = np.array([[ 1,  2,  1],
                               [ 0,  0,  0], 
                               [-1, -2, -1]])
    
    VerticalMask = np.array([[-1,  0,  1],
                             [-2,  0,  2], 
                             [-1,  0,  1]])
    
    DiagonalRightMask = np.array([[ 0,  1,  2],
                                  [-1,  0,  1], 
                                  [-2, -1,  0]])
    
    DiagonalLeftMask = np.array([[-2, -1,  0],
                                 [-1,  0,  1], 
                                 [ 0,  1,  2]])
    
    threads = []
    threads.append(threading.Thread(target=apply_mask, args=(image, HorizontalMask, Horizontal, edgeThreshold)))
    threads.append(threading.Thread(target=apply_mask, args=(image, VerticalMask, Vertical, edgeThreshold)))
    threads.append(threading.Thread(target=apply_mask, args=(image, DiagonalRightMask, DiagonalRight, edgeThreshold)))
    threads.append(threading.Thread(target=apply_mask, args=(image, DiagonalLeftMask, DiagonalLeft, edgeThreshold)))

    # Start threads
    for thread in threads:
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Final Edge Detection
    for i in range(final.shape[0]):
        for j in range(final.shape[1]):
            if Horizontal[i][j][0] == 255 or Vertical[i][j][0] == 255 or DiagonalRight[i][j][0] == 255 or DiagonalLeft[i][j][0] == 255:
                final[i][j] = [255, 0, 0]

    # draw the line on image
    for i in range(final.shape[0]):
        for j in range(final.shape[1]):
            if final[i][j][0] == 255:
                image[i][j] = [255, 0, 0]

    cv2.imwrite("final_image.jpg", image)

    


def process_image():
    global image_processing

    height, width, channels = image_processing.shape

    if image_processing is None:
        print("이미지가 없습니다.")

    print("이미지 전처리 버튼이 눌렸습니다.")
    print("이미지 처리하는 중...")
    # Display a message box indicating that image processing is in progress
    progress_msg = "이미지 처리 중입니다..."
    # progress_box = messagebox.showinfo("Processing", progress_msg)

    hsv_image = convert_to_hsv(image_processing)

    print("enhancing 처리 중...")
    enhance_image(image_processing)
    enhanced = cv2.imread("enhanced_image.jpg")
    print("enhancing 완료...")

    print("blur 처리 중...")
    blur_image(enhanced)
    blurred = cv2.imread("blurred.jpg")
    print("blur 완료...")

    print("deleting gum 처리 중...")
    delete_gum(enhanced)
    gum_deleted = cv2.imread("image_without_gum.jpg")
    print("deleting gum 완료...")

    # print("binarize 처리 중...")
    # binarize_image(gum_deleted)
    # binarized = cv2.imread("binarized.jpg")
    # print("binarize 완료...")

    # print("edge detection 처리 중...")
    # edge_detection(gum_deleted)
    # print("edge detection 완료...")

    file_path = 'image_without_gum.jpg'  # 파일 선택 다이얼로그 열기

    if file_path:
        print("이미지 전처리가 완료되었습니다.", file_path)

        # 이미지를 화면에 표시
        image = Image.open(file_path)
        image = image.resize((width, height))
        image = ImageTk.PhotoImage(image)
        label.config(image=image)
        label.image = image
        label.pack()

    detect_button.config(state=tk.NORMAL)

    # Close the message box after processing is complete
    # messagebox.showinfo("Processing Complete", "이미지 처리가 완료되었습니다.")
    root.lift()  # Bring the main window to the front
    detect_button.config(state=tk.NORMAL)




def detect_image():
    global image_processing

    img_path = 'enhanced_image.jpg'
    
    if img_path:
        img = image.load_img(img_path, target_size=(64, 64))
        img_array = image.img_to_array(img) #배열화
        img_array = np.expand_dims(img_array, axis=0) #(1,64,64,3) 64*64 크기의 RGB
        img_array /= 255.0 #정규화
        #어레이로 변환한거 이제 모델 넘기기

        # 모델을 사용하여 예측
        predictions = loaded_model.predict(img_array)

        # 충치 가능성 출력
        print(predictions)

        if predictions[0, 0] < 0.5:
            # Cavity인 경우
            print("Cavity!")
        else:
            # No Cavity인 경우
            print("No Cavity!")



def load_image():
    global image_processing
    file_path = filedialog.askopenfilename()  # 파일 선택 다이얼로그 열기

    if file_path:
        print("이미지를 불러왔습니다:", file_path)

        # Open the image and resize if necessary
        img = Image.open(file_path)

        # Check if resizing is needed
        if img.width > max_image_size[0] or img.height > max_image_size[1]:
            img.thumbnail(max_image_size)

        img = ImageTk.PhotoImage(img)
        print("파일 위치 : " + file_path)
        image_processing = cv2.imread(file_path)

        label.config(image=img)
        label.image = img
        label.pack()

        # 처리 버튼 및 판별 버튼 활성화
        process_button.config(state=tk.NORMAL)
        
        # Calculate the window size with a margin
        margin = 150
        window_width = min(img.width(), max_image_size[0]) + margin
        window_height = min(img.height(), max_image_size[1]) + margin
        root.geometry(f"{window_width}x{window_height}")

# Tkinter 창 생성
root = tk.Tk()
root.title("cavity detection")

#처음 window 사이즈 설정
root.geometry("400x400")

# 이미지 표시 레이블 생성
label = tk.Label(root)
label.pack()

# 이미지 불러오기 버튼 생성
load_button = tk.Button(root, text="get image", command=load_image)
load_button.pack(pady=10)

# 이미지 처리 버튼 생성 (비활성화 상태로 시작)
process_button = tk.Button(root, text="image processor", command=process_image, state=tk.DISABLED)
process_button.pack(pady=10)

# 이미지 처리 버튼 생성 (비활성화 상태로 시작)
detect_button = tk.Button(root, text="detect cavity", command=detect_image, state=tk.DISABLED)
detect_button.pack(pady=10)

# Tkinter 이벤트 루프 시작
root.mainloop()