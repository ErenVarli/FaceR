import os, sys
import face_recognition
import cv2
import numpy as np
import math
from gtts import gTTS
import json
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

def load_users():
    with open('users.json', 'r') as file:
        data = json.load(file)

    return data

def isWanted(user_id):
    users = load_users()
    
    for user in users:
        if user["user_id"] == user_id:
            if user["wanted"] == True:
                return True

def get_user_data(attribute, user_id):
    users = load_users()

    for user in users:
        if user["user_id"] == user_id:
            return user[attribute]
    return "Unknown"

def face_scan(user_id):
        if isWanted(user_id):
            message = f"{get_user_data("full_name", user_id)} detected. Warning: this people is wanted by the FBI."
        else: 
            message = f"{get_user_data("full_name", user_id)} detected."

        speech_voice(message)

def displayed_data(frame, id, confidence, emotion, left, bottom):
    first_name   =  "First Name: " + get_user_data("first_name", id)
    last_name    =  "Last Name: "  + get_user_data("last_name", id)
    full_name    =  "Full Name: "  + get_user_data("full_name", id)
    eye_color    =  "Eye Color: "  + get_user_data("eye_color", id)
    height       =  "Height: "     + get_user_data("height", id)
    sex          =  "Sex: "        + get_user_data("sex", id)
    description  =  "Description: " + get_user_data("description", id)

    square_full_name = get_user_data("full_name", id)
    wantedMsg    = "WARNING: WANTED BY THE FBI"

    FONT_COLOR_WHITE = (255, 255, 255)

    cv2.putText(frame, (square_full_name + " (" + confidence + ")"), (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, FONT_COLOR_WHITE, 1)
    cv2.putText(frame, first_name , (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, FONT_COLOR_WHITE, 1)
    cv2.putText(frame, last_name , (10, 60), cv2.FONT_HERSHEY_DUPLEX, 0.8, FONT_COLOR_WHITE, 1)
    cv2.putText(frame, full_name , (10, 90), cv2.FONT_HERSHEY_DUPLEX, 0.8, FONT_COLOR_WHITE, 1)
    cv2.putText(frame, eye_color , (10, 120), cv2.FONT_HERSHEY_DUPLEX, 0.8, FONT_COLOR_WHITE, 1)
    cv2.putText(frame, height , (10, 150), cv2.FONT_HERSHEY_DUPLEX, 0.8, FONT_COLOR_WHITE, 1)
    cv2.putText(frame, sex , (10, 180), cv2.FONT_HERSHEY_DUPLEX, 0.8, FONT_COLOR_WHITE, 1)
    cv2.putText(frame, description , (10, 210), cv2.FONT_HERSHEY_DUPLEX, 0.8, FONT_COLOR_WHITE, 1)
    cv2.putText(frame, ("Emotion: " + emotion), (10, 240), cv2.FONT_HERSHEY_DUPLEX, 0.8, FONT_COLOR_WHITE, 1)

    if isWanted(id):
        cv2.putText(frame, wantedMsg, (10, 270), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

def emotion_detection(frame, sc):
    smiles = sc.detectMultiScale(frame, 1.8, 40)
    if len(smiles) > 0:
        return True
    else:
        return False

def displayed_face_square(id, frame, top, right, bottom, left):
        if isWanted(id):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)
        else:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), -1)

def start_voice(read, file):
    audio_create = gTTS(text=read, lang="en", slow=False)
    audio_create.save(file)
    play_sound(file)

def play_sound(file):
    pygame.mixer.init()
    pygame.mixer.music.load(file)
    pygame.mixer.music.play()

def speech_voice(read):
    USER_AUDIO_FILE = "speech.mp3"
    read = read
    start_voice(read, USER_AUDIO_FILE)

def cap_settings(video_capture):
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not video_capture.isOpened():
            sys.exit("[ERR] - Problem with video source..")
        else: 
            return True

def face_confidence(face_distance, face_match_treshold=0.6):
    range = 1.0 - face_match_treshold
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_treshold:
        return str(round(linear_val * 100, 2)) + "%"
    else:
        value = (
            linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))
        ) * 100
        return str(round(value, 2)) + "%"

class FaceRecognition:
    face_locations        = []
    face_encodings        = []
    face_ids              = []
    known_face_encodings  = []
    known_face_id         = []
    process_current_frame = True
    new_id                = ""
    user_emotion          = ""
    video_capture         = cv2.VideoCapture(0)
    smile_cascade         = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml') 

    def set_user_emotion(self, emotion):
        self.user_emotion = emotion

    def get_user_emotion(self):
        return self.user_emotion

    def __init__(self):
        self.encode_faces()

    def encode_faces(self):
        for image in os.listdir("faces"):
            face_image = face_recognition.load_image_file(f"faces/{image}")
            face_encoding = face_recognition.face_encodings(face_image)[0]
            self.known_face_encodings.append(face_encoding)
            self.known_face_id.append(image)

    def run_recognition(self):
        if cap_settings(self.video_capture):
            print("[INFO] - CAMERA STATUS : ACTIVE")
            while True:
                ret, frame = self.video_capture.read()

                if self.process_current_frame:
                    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                    rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

                    self.face_locations = face_recognition.face_locations(rgb_small_frame)
                    self.face_encodings = face_recognition.face_encodings(
                        rgb_small_frame, self.face_locations
                    )

                    self.face_ids = []

                    for face_encoding in self.face_encodings:
                        matches = face_recognition.compare_faces(
                            self.known_face_encodings, face_encoding
                        )
                        id = "--"
                        confidence = "-"

                        face_distances = face_recognition.face_distance(
                            self.known_face_encodings, face_encoding
                        )
                        best_match_index = np.argmin(face_distances)

                        if matches[best_match_index]:
                            id = self.known_face_id[best_match_index]
                            confidence = face_confidence(face_distances[best_match_index])

                        self.face_ids.append(f"{id}")

                        if self.new_id != id and id != "--":
                            face_scan(id)
                            self.new_id = id

                self.process_current_frame = not self.process_current_frame

                for (top, right, bottom, left), id in zip(self.face_locations, self.face_ids):
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    if emotion_detection(frame, self.smile_cascade):
                        self.set_user_emotion("Positive")
                    else: 
                        self.set_user_emotion("Neutral")

                    displayed_face_square(id, frame, top, right, bottom, left)
                    displayed_data(frame, id, confidence, self.get_user_emotion(), left, bottom)

                cv2.imshow("Face Recognition", frame)
                
                if cv2.waitKey(1) == ord("x"):
                    break

        self.video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    fr = FaceRecognition()
    fr.run_recognition()


# Author : Eren Varli - FaceR - https://github.com/ErenVarli