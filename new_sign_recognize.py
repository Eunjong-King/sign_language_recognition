import numpy as np
import cv2
import mediapipe as mp
import joblib
import math
from files.unicode import join_jamos

def get_label():
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
    d23 = math.sqrt((d8.x * w - d12.x * w) ** 2 + (d8.y * h - d12.y * h) ** 2)
    d34 = math.sqrt((d16.x * w - d12.x * w) ** 2 + (d16.y * h - d12.y * h) ** 2)
    feature_list.append(d23 / d34 - 1)
    feature_list = np.round(feature_list, decimals=5)
    C = dic[kn.predict([feature_list])[0]]

    return C

def print_hangul(ch, previous_ch, my_word):
    checker1 = 0
    checker2 = 0
    if ch != previous_ch:
        if ch in ja:  # 자음입력
            for i in range(0, 11):
                if ch == jong[i][1] and previous_ch == jong[i][0]:
                    my_word = my_word[:-1]
                    my_word += jong[i][2]
                    checker1 = 1
                    checker2 = 1
                    break
            if checker1 == 0:  # 자음특수한 경우가 아닐때
                checker2 = 0
                my_word += ch
            previous_ch = ch
            print(join_jamos(my_word))
            checker1 = 0
        elif ch in mo:  # 모음입력
            if previous_ch == 'ㅗ' and ch in mo2:  # ㅗ였고 ㅏorㅐ이면
                my_word = my_word[:-1]
                previous_ch = ch
                ch = mo4[mo2.index(ch)]
            elif previous_ch == 'ㅜ' and ch in mo3:  # ㅜ였고 ㅓorㅔ이면
                my_word = my_word[:-1]
                previous_ch = ch
                ch = mo4[mo3.index(ch) + 2]
            else:  # 그냥 모음
                if checker2 == 1:
                    l = my_word[-1]
                    my_word = my_word[:-1]
                    for i in range(0, 11):
                        if jong[i][2] == l:
                            my_word += jong[i][0]
                            my_word += jong[i][1]
                            break
                    previous_ch = ch
                    checker2 = 0
                else:
                    previous_ch = ch
            my_word += ch
            print(join_jamos(my_word))
        else:  # a, b, c or d입력
            checker2 = 0
            if ch == 'a':
                if previous_ch in ssang:
                    my_word = my_word[:-1]
                    my_word += chr(ord(previous_ch) + 1)
            elif ch == 'c':
                my_word = my_word[:-1]
            elif ch == 'd':
                my_word += ' '
            print(join_jamos(my_word))
    return my_word, previous_ch


kn = joblib.load('files/ML-model.pkl')
cap = cv2.VideoCapture(0)
w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
dot_list = [4, 8, 12, 14, 16, 18, 20]
dic = {'q':'ㅂ', 'w':'ㅈ', 'e':'ㄷ', 'r':'ㄱ', 't':'ㅅ', 'y':'ㅛ', 'u':'ㅕ', 'i':'ㅑ', 'o':'ㅐ', 'p':'ㅔ',
       'a':'ㅁ', 's':'ㄴ', 'd':'ㅇ', 'f':'ㄹ', 'g':'ㅎ', 'h':'ㅗ', 'j':'ㅓ', 'k':'ㅏ', 'l':'ㅣ',
       'z':'ㅋ', 'x':'ㅌ', 'c':'ㅊ', 'v':'ㅍ', 'b':'ㅠ', 'n':'ㅜ', 'm':'ㅡ'}
my_char = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ', 'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ',
           'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ', 'ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅢ', 'ㅚ', 'ㅟ', 'a', 'b', 'c']
ja = ['ㄱ','ㄴ','ㄷ','ㄹ','ㅁ','ㅂ','ㅅ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
mo = ['ㅏ','ㅑ','ㅓ','ㅕ','ㅗ','ㅛ','ㅜ','ㅠ','ㅡ','ㅣ','ㅐ','ㅒ','ㅔ','ㅖ','ㅢ','ㅚ','ㅟ']
mo1 = ['ㅗ', 'ㅜ']
mo2 = ['ㅏ', 'ㅐ']
mo3 = ['ㅓ', 'ㅔ']
mo4 = ['ㅘ', 'ㅙ', 'ㅝ', 'ㅞ']
ssang = ['ㄱ','ㄷ','ㅂ','ㅅ','ㅈ']
jong = [['ㄱ', 'ㅅ', 'ㄳ'], ['ㄴ', 'ㅈ', 'ㄵ'], ['ㄴ', 'ㅎ', 'ㄶ'], ['ㄹ', 'ㄱ', 'ㄺ'], ['ㄹ', 'ㅁ', 'ㄻ'],
        ['ㄹ', 'ㅂ', 'ㄼ'], ['ㄹ', 'ㅅ', 'ㄽ'], ['ㄹ', 'ㅌ', 'ㄾ'], ['ㄹ', 'ㅍ', 'ㄿ'], ['ㄹ', 'ㅎ', 'ㅀ'],
        ['ㅂ', 'ㅅ', 'ㅄ']]
my_word = ''
dump_list = []
previous_ch = ''
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
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                C = get_label()
                dump_list.append(C)
                if len(dump_list) > 30 :
                    ch = max(dump_list)
                    my_word, previous_ch = print_hangul(ch, previous_ch, my_word)
                    dump_list = []
        cv2.imshow('mouse and keyboard', image)
        exit_code = cv2.waitKey(5)
        if exit_code == ord('q') or exit_code == 27:
            break

cap.release()