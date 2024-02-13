Outline of program
Program, through a graphical user interface (GUI), enables users to input oral images.
 Upon loading the oral image, the program performs image preprocessing using the functions of enhancing (intensifying the color of the gums), blur (noise reduction for precise discrimination), and deleting gum (removing the gums). 
Subsequently, it utilizes an implemented machine learning model to determine whether the corresponding tooth has cavities or not. 
Through this process, users can easily assess oral health by loading oral images via the GUI and checking the results for the respective teeth.

Design of program
We wanted to separate the gum part and teeth part of the original oral image, because when we try to detect a cavity, the gum part of the image is not very useful. So we tried to delete the gum part of the image to get a higher accuracy for detection of cavities. We performed the following processes to reach our goal.

<Image processing>
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

The “load_image” function serves as the entry point for a program that loads oral images for classification. Through this function, images are loaded and displayed on a Tkinter window.

def process_image():
    global image_processing
    height, width, channels = image_processing.shape
    if image_processing is None:
        print("이미지가 없습니다.")
    print("이미지 전처리 버튼이 눌렸습니다.")
    print("이미지 처리하는 중...")
    progress_msg = "이미지 처리 중입니다..."
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
    root.lift()  # Bring the main window to the front
    detect_button.config(state=tk.NORMAL)
Using images loaded by “load_image()” functions, to discern cavities, a sequence of preprocessing steps is performed using theconvert_to_hsv(), enhance_image(), blur_image(), delete_gum() functions. 

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
“convert_to_hsv” function converts an image received in RGB format to HSV. HSV separates color and brightness independently compared to RGB, making it simpler to select specific color ranges or apply filtering. This conversion is performed to facilitate subsequent operations.

def enhance_image(image):
    # 이미지를 HSV로 변환
    hsv_image = convert_to_hsv(image)
    # 잇몸 부분(빨간색) 감지
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    red_mask = inRange(hsv_image, lower_red, upper_red)
    lower_pink = np.array([150, 50, 50])
    upper_pink = np.array([179, 255, 255])
    pink_mask = inRange(hsv_image, lower_pink, upper_pink)
    mask_gum = red_mask + pink_mask
    gum_part = cv2.bitwise_and(image, image, mask=mask_gum)
    gum_part = np.clip(gum_part + [20, 0, 30], 0, 255)  # 빨간색 강도 증가
    gum_part = cv2.bitwise_and(gum_part, gum_part, mask=mask_gum)
    tooth_part = cv2.bitwise_and(image, image, mask=~mask_gum)
    tooth_part = cv2.bitwise_and(tooth_part, tooth_part, mask=~mask_gum)
    gum_part = gum_part.astype(np.uint8)
    tooth_part = tooth_part.astype(np.uint8)
    enhanced_image = cv2.bitwise_xor(gum_part, tooth_part)
    cv2.imwrite("enhanced_image.jpg", enhanced_image)
convert_to_hsv() function is used to detect and emphasize the gum area in the oral image, which has been converted to HSV format. To detect the gum area, a mask is created for the regions corresponding to red and pink. The image is then separated into gum and tooth regions based on this mask. The intensity of the red color in the gum area is increased, and the enhanced gum and tooth regions are merged back together. The final result is saved as 'enhanced_image.jpg'.

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
Image is blurred using a 3x3 blur mask to reduce fine details and create a smoother appearance by averaging the surrounding colors and intensities. The result of applying this blur operation is saved as 'blurred.jpg'.

def delete_gum(image):
    hsv_image = convert_to_hsv(image)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    red_mask = inRange(hsv_image, lower_red, upper_red)
    lower_pink = np.array([150, 50, 50])
    upper_pink = np.array([179, 255, 255])
    pink_mask = inRange(hsv_image, lower_pink, upper_pink)
    mask_gum = red_mask + pink_mask
    # 잇몸 부분을 검은색으로 칠하기
    image_without_gum = cv2.bitwise_and(image, image, mask=~mask_gum)
    # 결과 저장
    cv2.imwrite("image_without_gum.jpg", image_without_gum)
The regions in the image corresponding to red and pink colors are filled with black. After removing the gum from this modified image, the result is saved as 'image_without_gum.jpg'.

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
“process_image()” function provides the results regarding whether cavities are present in the processed image. It utilizes the trained cavity detection model, 'model.h5', which has been trained using the “Image_load_img” method. The function proceeds with the prediction using the model and prints the prediction result. If the predicted value is less than 0.5, it outputs "Cavity!" otherwise, it prints "No cavity!"


<Model>
Wrote the model code by referencing the open-source project at “https://github.com/teeth-check”
# 각 이미지에 대한 예측 및 분류
for img_path in image_paths:
    # 이미지 불러오기 및 전처리
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # 모델 학습 시 사용한 rescaling을 적용
    # 모델에 이미지 주입 및 예측 수행
    predictions = model.predict(img_array)
    # 예측 결과에 따라 분류
    if predictions[0, 0] < 0.5:
        # Cavity인 경우
        shutil.copy(img_path, os.path.join(output_folder_cavity, os.path.basename(img_path)))
    else:
        # No Cavity인 경우
        shutil.copy(img_path, os.path.join(output_folder_no_cavity, os.path.basename(img_path)))

After processing the aforementioned preprocessing-related code, it takes dental images and classifies them into decayed and normal teeth. This comprehensive preprocessing ensures that the dental images are optimized for the decay detection model, enhancing the accuracy of classification between decayed and normal teeth while also excluding unnecessary background information such as gums.

def build_model(dropout_rate=0.0):
    model = Sequential()
    model.add(Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),padding='same',input_shape=(128,128,3),activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(filters=64, kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(filters=64, kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(512,activation='relu'))
    model.add(Dropout(dropout_rate)) 
    model.add(Dense(2,activation='softmax'))
    model.compile( 
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        )    
    return model

Convolutional Neural Network (CNN) model using the Keras Sequential API for image classification tasks. The model consists of three convolutional layers followed by max-pooling layers to extract features, a flattening layer to convert the data into a 1D array, a dense layer with 512 neurons and ReLU activation, a dropout layer for regularization, and a final dense layer with two neurons for binary classification using softmax activation. The model is compiled with the Adam optimizer, categorical crossentropy loss function, and accuracy as the evaluation metric. 
