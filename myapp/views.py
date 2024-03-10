import base64
import os
from threading import Timer
import numpy as np
from django.shortcuts import render
from django.contrib import messages
from django.http import HttpResponse, JsonResponse
import cv2
from django.shortcuts import render
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from django.templatetags.static import static

# Read in the cascade classifiers for face and eyes
face_cascade = cv2.CascadeClassifier('../DATA / haarcascades / haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('../DATA / haarcascades / haarcascade_eye.xml')
from .forms import InputForm
def index(request):
    return render(request,"index_template.html")
def capture(request):
    import cv2
    messages.info(request, 'Make your face expression')
    videoCaptureObject = cv2.VideoCapture(0)
    result = True
    while (result):
        ret, frame = videoCaptureObject.read()
        cv2.imwrite("E://iqra_ML_project//Facepro (2)//Facepro//mypro//static//images//img1.jpg", frame)
        result = False
    videoCaptureObject.release()
    cv2.destroyAllWindows()
    cascPath = "E://iqra_ML_project//Facepro (2)//Facepro//mypro//opencv//haarcascade_frontalface_default.xml"
    i = 0
    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascPath)
    video_capture = cv2.VideoCapture(0)
    emotion_num_map = []
    while (True):
    # Capture frame-by-frame
        ret, frame = video_capture.read()
        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            frame,
            scaleFactor=1.1,
            minNeighbors=2,
            minSize=(2, 2),
            # flags = cv2.CV_HAAR_SCALE_IMAGE
        )
        # print faces found ......
        facesNumber ="Found {0} faces!".format(len(faces))
        print(facesNumber)
        # test
    return render(request, 'capture_emotion.html', {'p': facesNumber})
    # return render(request,'webcam.html')
def feedback(request):
    return render(request,"feedback.html")
def welcome(request):
    message = 'Django'
    return render(request, 'welcome.html', {'p': message})
    # for each emotion.The code below will iterate through all seven emotion classes and plot the randomly selected 1 image from each class.
# def plotimage(request):
#     plt.figure(0, figsize=(16, 10))
#     for i in range(7):
#         plt.subplot(2, 4, i + 1)
#         image = save_images()
#         plt.imshow(image)
        # plt.title(emotion_num_map[i])
def preprocess(input_data):
    input_images = np.zeros(shape=(input_data.shape[0], 48, 48))
    for i, row in enumerate(input_data.index):
        image = np.fromstring(input_data.loc[row, ' pixels'], dtype=int, sep=' ')
        image = np.reshape(image, (48, 48))
        input_images[i] = image
    input_images = input_images.reshape((input_images.shape[0], 48, 48, 1))
    input_images = input_images.astype('float32')/255
    input_labels = np.array(list(map(int, input_data['emotion'])))
    image_labels = to_categorical(input_labels, 7)
    return  input_images, image_labels

def html1(request):
    context = {}
    context['form'] = InputForm()
    return render(request, "index2.html", context)
    #img_path = "E:/pythonProject/Facepro (2)/Facepro/mypro/static/images/anger.png"
def detectemotion(request):
    from deepface import DeepFace
    import cv2

    # Load the image
    img_path = "E:/iqra_ML_project/Facepro (2)/Facepro/mypro/static/images/img1.jpg"
    img = cv2.imread(img_path)

    # Perform emotion analysis
    result = DeepFace.analyze(img_path, actions=['emotion'])
    print(" -> ",result)
    # Get emotion prediction
    # emotion = result['emotion']
    dominant_emotion = max(result[0]['emotion'].items(), key=lambda x: x[1])[0]
    accuracy = None# Get predicted emotion
    dominant_emotion = max(result[0]['emotion'].items(), key=lambda x: x[1])[0]
    print("dominant emotion : ",dominant_emotion)
    # Define the actual emotion (if available)
    actual_emotion = "Happy"  # Replace with the actual emotion label if available

    if actual_emotion:
        accuracy = 1 if dominant_emotion == actual_emotion else 0
        # print("Accuracy:", accuracy)
    # Print or use the result
    print("Dominant Emotion:", dominant_emotion)
    # Get face coordinates
    face_coordinates = result[0]['region']

    # Extract individual coordinates
    x = face_coordinates['x']
    y = face_coordinates['y']
    w = face_coordinates['w']
    h = face_coordinates['h']

    # Print or use the coordinates
    print("Face Coordinates: x={}, y={}, w={}, h={}".format(x, y, w, h))
    # Draw a rectangle around the detected face
    cv2.rectangle(img, (x-30, y-30), (x + w+100, y + h+100), (255, 0, 0), 2)

    # Display the result
    cv2.imshow('Emotion Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Get the coordinates of the face
    # face_coordinates = result['region']

    # Draw a rectangle around the detected face
    # x, y, w, h = face_coordinates
    # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Put the emotion label on the image
    cv2.putText(img, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the result
    cv2.imshow('Emotion Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return render(request,"index_template.html")



def loadhtml(request):
    return render(request,'index_template.html')

def about(request):
    return render(request,'about.html')

def prepare_dataset(request):
    import numpy as np
    import pandas as pd
    from PIL import Image
    from tqdm import tqdm
    import os

    # convert string to integer
    def atoi(s):
        n = 0
        for i in s:
            n = n * 10 + ord(i) - ord("0")
        return n

    # making folders
    outer_names = ['test', 'train']
    inner_names = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'disgust','neutral']
    os.makedirs('data', exist_ok=True)
    for outer_name in outer_names:
        os.makedirs(os.path.join('data', outer_name), exist_ok=True)
        for inner_name in inner_names:
            os.makedirs(os.path.join('data', outer_name, inner_name), exist_ok=True)

    # to keep count of each category
    angry = 0
    disgusted = 0
    fearful = 0
    happy = 0
    sad = 0
    surprised = 0
    neutral = 0
    angry_test = 0
    disgusted_test = 0
    fearful_test = 0
    happy_test = 0
    sad_test = 0
    surprised_test = 0
    neutral_test = 0

    df = pd.read_csv('./fer2013.csv')
    mat = np.zeros((48, 48), dtype=np.uint8)
    print("Saving images...")

    # read the csv file line by line
    for i in tqdm(range(len(df))):
        txt = df['pixels'][i]
        words = txt.split()

        # the image size is 48x48
        for j in range(2304):
            xind = j // 48
            yind = j % 48
            mat[xind][yind] = atoi(words[j])

        img = Image.fromarray(mat)

        # train
        if i < 28709:
            if df['emotion'][i] == 0:
                img.save('train/angry/im' + str(angry) + '.png')
                angry += 1
            elif df['emotion'][i] == 1:
                img.save('train/disgusted/im' + str(disgusted) + '.png')
                disgusted += 1
            elif df['emotion'][i] == 2:
                img.save('train/fearful/im' + str(fearful) + '.png')
                fearful += 1
            elif df['emotion'][i] == 3:
                img.save('train/happy/im' + str(happy) + '.png')
                happy += 1
            elif df['emotion'][i] == 4:
                img.save('train/sad/im' + str(sad) + '.png')
                sad += 1
            elif df['emotion'][i] == 5:
                img.save('train/surprised/im' + str(surprised) + '.png')
                surprised += 1
            elif df['emotion'][i] == 6:
                img.save('train/neutral/im' + str(neutral) + '.png')
                neutral += 1

        # test
        else:
            if df['emotion'][i] == 0:
                img.save('test/angry/im' + str(angry_test) + '.png')
                angry_test += 1
            elif df['emotion'][i] == 1:
                img.save('test/disgusted/im' + str(disgusted_test) + '.png')
                disgusted_test += 1
            elif df['emotion'][i] == 2:
                img.save('test/fearful/im' + str(fearful_test) + '.png')
                fearful_test += 1
            elif df['emotion'][i] == 3:
                img.save('test/happy/im' + str(happy_test) + '.png')
                happy_test += 1
            elif df['emotion'][i] == 4:
                img.save('test/sad/im' + str(sad_test) + '.png')
                sad_test += 1
            elif df['emotion'][i] == 5:
                img.save('test/surprised/im' + str(surprised_test) + '.png')
                surprised_test += 1
            elif df['emotion'][i] == 6:
                img.save('test/neutral/im' + str(neutral_test) + '.png')
                neutral_test += 1

    print("Done!")
