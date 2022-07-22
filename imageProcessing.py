import random
import cv2
import glob
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn import svm
from sklearn.naive_bayes import GaussianNB, CategoricalNB


def svmClass(features, output, featuresO, testO):
    clf = svm.SVC(kernel='linear')
    clf.fit(features, output)
    outA = clf.predict(featuresO)
    acc = accuracy_score(testO, outA) * 100
    pre = precision_score(testO, outA, average='micro')
    rec = recall_score(testO, outA, average='micro')
    return pre, rec, acc


def naiveBayes(features, output, featuresO, testO):
    cnb = CategoricalNB()
    cnb.fit(features, output)
    y_pred = cnb.predict(featuresO)
    acc = accuracy_score(testO, y_pred) * 100
    pre = precision_score(testO, y_pred, average='micro')
    rec = recall_score(testO, y_pred, average='micro')
    return pre, rec, acc


def logClass(features, output, featuresO, testO):
    classifier = LogisticRegression(max_iter=4500)
    classifier.fit(features, output)
    outA = classifier.predict(featuresO)
    acc = accuracy_score(testO, outA) * 100
    pre = precision_score(testO, outA, average='micro')
    rec = recall_score(testO, outA, average='micro')
    return pre, rec, acc


def flatten(imageNames):
    readImagesGray = []
    readImagesBinary = []
    readImagesRGB = []
    readImagesGrayCanny = []
    readImagesBinaryCanny = []
    readImagesRGBCanny = []

    images = imageNames

    for image in images:
        imgRGB = cv2.imread(image)
        imgRGB = cv2.resize(imgRGB, (200, 200))
        imgRGB = cv2.resize(imgRGB, (0, 0), fx=0.25, fy=0.25)

        imgBlurRGB = cv2.GaussianBlur(imgRGB, (3, 3), 0)
        edges = cv2.Canny(image=imgBlurRGB, threshold1=4, threshold2=100)  # Canny Edge Detection
        readImagesRGBCanny.append(edges)
        readImagesRGB.append(edges)

        imgG = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        imgG = cv2.resize(imgG, (200, 200))
        imgG = cv2.resize(imgG, (0, 0), fx=0.25, fy=0.25)

        readImagesGray.append(imgG)
        imgBlurGray = cv2.GaussianBlur(imgG, (3, 3), 0)
        edgesGray = cv2.Canny(image=imgBlurGray, threshold1=4, threshold2=100)  # Canny Edge Detection
        readImagesGrayCanny.append(edgesGray)

        img = cv2.imread(image, 2)
        img = cv2.resize(img, (200, 200))
        img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        r, threshold = cv2.threshold(img, 149, 255, cv2.THRESH_BINARY)
        readImagesBinary.append(threshold)
        imgBlurBinary = cv2.GaussianBlur(img, (3, 3), 0)
        edgesBinary = cv2.Canny(image=imgBlurBinary, threshold1=4, threshold2=100)  # Canny Edge Detection
        readImagesBinaryCanny.append(edgesBinary)

    lenofimage = len(images)
    readImagesGrayCanny = np.array(readImagesGrayCanny)
    readImagesRGBCanny = np.array(readImagesRGBCanny)
    readImagesBinaryCanny = np.array(readImagesBinaryCanny)

    flattenedRGB = np.array(readImagesRGBCanny).reshape(lenofimage, -1)
    flattenedGray = np.array(readImagesGrayCanny).reshape(lenofimage, -1)
    flattenedBinary = np.array(readImagesBinaryCanny).reshape(lenofimage, -1)
    # print(flattenedBinary.shape)

    return flattenedBinary, flattenedGray, flattenedRGB


def ReadFiles():
    directory = glob.glob(
        '/content/ASL_Alphabet_Dataset/asl_alphabet_train/*')
    imageNamesTrain = []
    labelOutputTrain = []
    for folder in directory:
        for file in glob.glob(folder + '/*.jpg'):
            # print(file)
            imageNamesTrain.append(file)

    imageNamesT = random.sample(imageNamesTrain, len(imageNamesTrain))
    imageTrain = imageNamesT  # [:100400]
    for image in imageTrain:
        labels = image.split("/")
        name = labels[4]
        labelOutputTrain.append(name)

    labelOutputTest = []
    imageNamesTest = []
    for file in glob.glob('/content/ASL_Alphabet_Dataset/asl_alphabet_test/*.jpg'):
        imageNamesTest.append(file)

    for image in imageNamesTest:
        labels = image.split("/")[-1]
        # print(labels)
        finLabel = labels.split("_")[0]
        print(finLabel)
        labelOutputTest.append(finLabel)
    TrainData = imageTrain
    TestData = imageNamesTest
    return TrainData, TestData, labelOutputTest, labelOutputTrain


imageNamesTrain, imageNamesTest, labelOutputTest, labelOutputTrain = ReadFiles()
trainBinary, trainGray, trainRGB = flatten(imageNamesTrain)
testB, testG, testRGB = flatten(imageNamesTest)

TrainGray = trainGray[:int(len(trainGray) * 0.6)]
validateGray = trainGray[int(len(trainGray) * 0.6):]
TrainBinary = trainBinary[:int(len(trainBinary) * 0.6)]
validateBinary = trainBinary[int(len(trainBinary) * 0.6):]
TrainRGB = trainRGB[:int(len(trainRGB) * 0.6)]
validateRGB = trainRGB[int(len(trainRGB) * 0.6):]

listTrainCases = [trainBinary, trainGray, trainRGB]
listTestCases = [testB, testG, testRGB]

print("Models Test:")
types = ["Binary", "Gray", "RGB"]

for i in range(3):

    print(f"SVM with {types[i]}:", svmClass(listTrainCases[i], labelOutputTrain, listTestCases[i], labelOutputTest))

    print(f"Logestic with {types[i]}:",
          logClass(listTrainCases[i], labelOutputTrain, listTestCases[i], labelOutputTest))

    print(f"Naive with {types[i]}:", naiveBayes(listTrainCases[i], labelOutputTrain, listTestCases[i], labelOutputTest))

"""
Best Models:
    Binary: SVM was the best model trained against binary images (93%)
    Gray: SVM was the best model trained against gray images (97%)
    RGB: Logestic was the best model trained against RGB images(95%)
    
Naive Bayes have accuracies varying from 62% to 75% against binary, gray, and RGB images using categoricalNB
Naive Bayes have accuracies varying from 67% to 71% against binary, gray, and RGB images using GuassianNB
"""
