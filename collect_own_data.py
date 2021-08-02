import cv2
import mediapipe as mp
import csv
from math import sqrt
import os

def write_csv(image, key_input):
    label = key_input
    feature_list = []
    mean_x = hand_landmarks.landmark[0].x  # x가 왼오 0이 왼 1이 오
    mean_y = hand_landmarks.landmark[0].y  # y가 위아래 0이 젤위 1이 젤아래
    min_x = w - 1;
    max_x = 0.0;
    min_y = h - 1;
    max_y = 0.0
    for i in range(0, 21):  # 요기부터
        hlm = hand_landmarks.landmark[i]
        if hlm.x * w > max_x:
            max_x = hlm.x * w
        if hlm.x * w < min_x:
            min_x = hlm.x * w
        if hlm.y * h > max_y:
            max_y = hlm.y * h
        if hlm.y * h < min_y:
            min_y = hlm.y * h
    for i in dot_list:
        hlm = hand_landmarks.landmark[i]
        feature_list.append(((hlm.x - mean_x) * w) / (max_x - min_x))
        feature_list.append((hlm.y - mean_y) * h / (max_y - min_y))
    d8 = hand_landmarks.landmark[8]
    d12 = hand_landmarks.landmark[12]
    d16 = hand_landmarks.landmark[16]
    d23 = sqrt((d8.x * w - d12.x * w) ** 2 + (d8.y * h - d12.y * h) ** 2)
    d34 = sqrt((d16.x * w - d12.x * w) ** 2 + (d16.y * h - d12.y * h) ** 2)
    feature_list.append(d23 / d34 - 1)
    feature_list.append((max_y - min_y) / (max_x - min_x) - 1)
    cv2.rectangle(image, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 0, 0), 2)
    feature_list.append(label)
    wr.writerow(feature_list)


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
temp_file = open('files/temp_data.csv', 'w', newline='')
wr = csv.writer(temp_file)
dot_list = [4, 8, 12, 14, 16, 18, 20]
w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
key_list = [-1, 27, 45, 61]
with mp_hands.Hands(min_detection_confidence=0.5,
                    min_tracking_confidence=0.999,
                    max_num_hands=1
                    ) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                key_input = cv2.waitKey(20)

                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                if key_input not in key_list:
                    print(key_input)
                    write_csv(image, key_input)
                    cv2.putText(image, chr(key_input), (0, 30), cv2.FONT_HERSHEY_PLAIN, 2,
                                (0, 0, 255), thickness=3)
                else:
                    cv2.putText(image, "press key to save / '-' for save, '+' for don't save", (0, 30), cv2.FONT_HERSHEY_PLAIN, 1,
                                (0, 0, 255), thickness=3)

        cv2.imshow('mouse and keyboard', image)
        exit_code = cv2.waitKey(5)
        if exit_code == 45:
            temp_file.close()
            if os.path.isfile("files/temp_data.csv"):
                os.remove("files/temp_data.csv")
            break
        elif exit_code == 61:
            temp_file.close()
            t = open("files/temp_data.csv")
            reader = csv.reader(t)
            f = open("files/dataset.csv", 'a', newline='', encoding='utf-8')
            wr = csv.writer(f)
            for row in reader:
                wr.writerow(row)
            f.close()
            t.close()
            if os.path.isfile("files/temp_data.csv"):
                os.remove("files/temp_data.csv")
            break

cap.release()