import numpy as np
import cv2
import mediapipe as mp
import joblib
import math
from files.unicode import join_jamos
from collections import Counter


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
    feature_list.append((max_y - min_y) / (max_x - min_x) - 1)
    feature_list = np.round(feature_list, decimals=5)
    C = label_char[kn.predict([feature_list])[0]]

    return C

def merge_hangul(ch, previous_ch, my_word):
    if ch == previous_ch:
        return my_word, ch
    # 무조건 전에 입력한거랑 달라야함 => ㄱ인식됐는데 다음 할게 헷갈리면 ㄱ만 200번 입력되는 대참사 막기 위함
    else:
        # ch가 자음일 때
        if ch in ja:
            # previous_ch도 자음이라서 만약에 합쳐질 수 있는 경우 ex) ㄱ + ㅅ => ㄳ
            for pre, now, res in merge_ja:
                if pre == previous_ch and now == ch:
                    my_word = my_word[:-1] + res
                    return my_word, res
            # 일반적인 경우
            my_word += ch
            return my_word, ch

        # ch가 모음일 때
        elif ch in mo:
            # ㅗ + ㅏ => ㅘ 처럼 합쳐지는 경우
            if previous_ch in mo:
                for pre, now, res in merge_mo:
                    if pre == previous_ch and now == ch:
                        my_word = my_word[:-1] + res
                        return my_word, res
            # ㄱ -> ㅏ -> ㄱ -> ㅅ -> ㅏ 인경우 "갃ㅏ"가 아닌 "각사"가되야함
            elif previous_ch in np.array(merge_ja)[:, 2]:
                for pre, now, res in merge_ja:
                    if res == previous_ch:
                        my_word = my_word[:-1] + pre + now + ch
                        return my_word, ch
            # 아무것도 아닌 경우
            my_word += ch
            return my_word, ch

        # ch가 특수기호일 때
        else:
            # 1일때 쌍자음으로 만들기, 2일때 재입력, 3일때 백스페이스, 4일때 스페이스
            if ch == '1':
                if previous_ch in ssang:
                    # 다행히도 쌍자음의 유니코드는 그냥 자음의 +1이다
                    my_word = my_word[:-1] + chr(ord(previous_ch) + 1)
            elif ch == '2':
                pass
            elif ch == '3':
                my_word = my_word[:-1]
            elif ch == '4':
                my_word += ' '

            return my_word, ch




kn = joblib.load('files/ML-model.pkl')
cap = cv2.VideoCapture(0)
w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
dot_list = [4, 8, 12, 14, 16, 18, 20]
label_char = ['ㄱ','ㄴ','ㄷ','ㄹ','ㅁ','ㅂ','ㅅ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ',
              'ㅏ','ㅑ','ㅓ','ㅕ','ㅗ','ㅛ','ㅜ','ㅠ','ㅡ','ㅣ','ㅐ','ㅒ','ㅔ','ㅖ','ㅢ','ㅚ','ㅟ',
              '1','2','3','4']
ja = ['ㄱ','ㄴ','ㄷ','ㄹ','ㅁ','ㅂ','ㅅ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
mo = ['ㅏ','ㅑ','ㅓ','ㅕ','ㅗ','ㅛ','ㅜ','ㅠ','ㅡ','ㅣ','ㅐ','ㅒ','ㅔ','ㅖ','ㅢ','ㅚ','ㅟ']
merge_mo = [['ㅗ', 'ㅏ', 'ㅘ'], ['ㅗ', 'ㅐ', 'ㅙ'], ['ㅜ', 'ㅓ', 'ㅝ'], ['ㅜ', 'ㅔ', 'ㅞ']]
ssang = ['ㄱ','ㄷ','ㅂ','ㅅ','ㅈ']
merge_ja = [['ㄱ', 'ㅅ', 'ㄳ'], ['ㄴ', 'ㅈ', 'ㄵ'], ['ㄴ', 'ㅎ', 'ㄶ'], ['ㄹ', 'ㄱ', 'ㄺ'], ['ㄹ', 'ㅁ', 'ㄻ'],
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
                if previous_ch != C:
                    dump_list.append(C)
                if len(dump_list) > 20:
                    ch = Counter(dump_list).most_common()[0][0]
                    my_word, previous_ch = merge_hangul(ch, previous_ch, my_word)
                    print(join_jamos(my_word))
                    dump_list = []
        cv2.imshow('mouse and keyboard', image)
        exit_code = cv2.waitKey(5)
        if exit_code == ord('q') or exit_code == 27:
            break

cap.release()