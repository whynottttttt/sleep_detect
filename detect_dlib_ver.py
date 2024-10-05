import cv2
import dlib
import numpy as np
from model import Net
import torch
from imutils import face_utils

#얼굴 이미지에서 눈으로 인식하여 crop한 2개 이미지에 대한 사이즈
IMG_SIZE = (34,26)
PATH = 'weights/classifier_weights_iter_50.pt'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dlib의 얼굴을 인식할 수 있는 detector와 predictor를 선언
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

#눈을 감았는지 떴는 지 판단하는 model을 불러옴
model = Net()
model.load_state_dict(torch.load(PATH, weights_only=True))
model.eval()

n_count = 0 # 프레임을 저장하는 변수

# 이미지에서 눈 부분을 자르는 함수
# face landmark 68
def crop_eye(img, eye_points):
  x1, y1 = np.amin(eye_points, axis=0) # 점을 통해 x1, x2, y1, y2의 좌표를 뽑음 / openCV에서 (0,0)에 해당하는 점은 왼쪽 맨 위
  x2, y2 = np.amax(eye_points, axis=0)
  cx, cy = (x1 + x2) / 2, (y1 + y2) / 2 # x의 중점은 cx, y의 중점은 cy

  w = (x2 - x1) * 1.2 # width 값 구하기
  h = w * IMG_SIZE[1] / IMG_SIZE[0] # height 값 구하기

  margin_x, margin_y = w / 2, h / 2 # 마진 값 구하기

# 마진 값을 고려하여 x,y 좌표를 구하기
  min_x, min_y = int(cx - margin_x), int(cy - margin_y)
  max_x, max_y = int(cx + margin_x), int(cy + margin_y)

  eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(int) # 마진 값을 고려하여 x,y 좌표를 구해주고 이를 통해 eye_rect 좌표를 만듦

  eye_img = gray[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]] # 좌표에 해당하는 이미지를 grayscale로 변경함

  return eye_img, eye_rect # 이미지와 방금 구한 eye_rect 좌표를 return

# model을 통해 예측하는 함수
def predict(pred):
  pred = pred.transpose(1, 3).transpose(2, 3) # 단순하게 들어오는 눈 이미지를 transpose하여 model input과 같은 size로 변경

  outputs = model(pred)

  pred_tag = torch.round(torch.sigmoid(outputs)) #model을 통해 output을 test code와 마찬가지로 round를 통해 값을 return

  return pred_tag

cap = cv2.VideoCapture(0)

# main에 해당하는 부분
while cap.isOpened():
  ret, img_ori = cap.read()

  if not ret:
    break

  img_ori = cv2.resize(img_ori, dsize=(0, 0), fx=0.5, fy=0.5) # cv2.resize를 통해 사이즈 조절

  img = img_ori.copy()
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # rgb image를 gray scale로 전환

  faces = detector(gray)  #detector를 통해 gray scale image에서 얼굴을 인식

  for face in faces:
    shapes = predictor(gray, face) #  predictor를 통해 face 좌표에서 landmark를 추정
    shapes = face_utils.shape_to_np(shapes)

  # 눈 부분만 crop을 하여 eye_img_l, eye_rect_l, eye_img_r, eye_rect_r에 이미지와 눈 좌표를 저장
    eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shapes[36:42])
    eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shapes[42:48])

  # 눈이 인식된 이미지를 model input size에 맞게 조절
    eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
    eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
    eye_img_r = cv2.flip(eye_img_r, flipCode=1)

    # 눈만 인식한 결과를 볼 때
    # cv2.imshow('l', eye_img_l)
    # cv2.imshow('r', eye_img_r)

    eye_input_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32)
    eye_input_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32)

  # 우리 이미지는 numpy배열이니깐 torch.from_numpy를 통해 tensor로 변환
    eye_input_l = torch.from_numpy(eye_input_l)
    eye_input_r = torch.from_numpy(eye_input_r)

  # 우리 model의 해당하는 predict를 통해 눈 감았는 지 판단
    pred_l = predict(eye_input_l)
    pred_r = predict(eye_input_r)


    if pred_l.item() == 0.0 and pred_r.item() == 0.0:
      n_count+=1

    else:
      n_count = 0

    # 100 프레임 동안 눈을 감고 있다면 wake_up이라는 문구가 나오도록 코드 작성
    if n_count > 100:
      cv2.putText(img,"Wake up", (120,160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)



    # visualize
    state_l = 'O %.1f' if pred_l > 0.1 else '- %.1f'
    state_r = 'O %.1f' if pred_r > 0.1 else '- %.1f'

    state_l = state_l % pred_l
    state_r = state_r % pred_r


    cv2.rectangle(img, pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(255,255,255), thickness=2)
    cv2.rectangle(img, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=(255,255,255), thickness=2)

    cv2.putText(img, state_l, tuple(eye_rect_l[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(img, state_r, tuple(eye_rect_r[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
  #
  # print(state_l)
  # print(state_r)
  cv2.imshow('result', img)
  # cv2.waitKey(0) # 정지된 화면 볼 때

  if cv2.waitKey(1) == ord('q'): # q 입력하면 프로그램 종료
    break

cap.release()
cv2.destroyAllWindows()
