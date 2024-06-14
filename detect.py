import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
import argparse
from threading import Thread
import tensorflow as tf
from mtcnn import MTCNN

def highlightFace(detector, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    results = detector.detect_faces(frame)
    faceBoxes = []
    for result in results:
        confidence = result['confidence']
        if confidence > conf_threshold:
            x, y, width, height = result['box']
            faceBoxes.append([x, y, x + width, y + height])
            cv2.rectangle(frameOpencvDnn, (x, y), (x + width, y + height), (0, 255, 0), 2)
    return frameOpencvDnn, faceBoxes

def preprocessFace(face):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    face = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
    return face

def processVideo(video_source):
    faceProto = "opencv_face_detector.pbtxt"
    faceModel = "opencv_face_detector_uint8.pb"
    ageProto = "age_deploy.prototxt"
    ageModel = "age_net.caffemodel"
    genderProto = "gender_deploy.prototxt"
    genderModel = "gender_net.caffemodel"

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']

    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)
    faceDetector = MTCNN()

    video = cv2.VideoCapture(video_source)
    padding = 20

    while cv2.waitKey(1) < 0:
        hasFrame, frame = video.read()
        if not hasFrame:
            cv2.waitKey()
            break

        resultImg, faceBoxes = highlightFace(faceDetector, frame)
        if not faceBoxes:
            print("No face detected")

        for faceBox in faceBoxes:
            face = frame[max(0, faceBox[1] - padding):
                         min(faceBox[3] + padding, frame.shape[0] - 1),
                         max(0, faceBox[0] - padding):
                         min(faceBox[2] + padding, frame.shape[1] - 1)]

            face = preprocessFace(face)
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            genderConfidence = genderPreds[0].max()
            print(f'Gender: {gender} (Confidence: {genderConfidence:.2f})')

            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
            ageConfidence = agePreds[0].max()
            print(f'Age: {age[1:-1]} years (Confidence: {ageConfidence:.2f})')

            cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Detecting age and gender", resultImg)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help='Path to image or video file. Leave empty for webcam.')
    args = parser.parse_args()

    video_source = args.image if args.image else 0
    video_thread = Thread(target=processVideo, args=(video_source,))
    video_thread.start()

if __name__ == "__main__":
    main()
