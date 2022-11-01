import torch
import numpy as np
import cv2
import pafy
from functools import reduce
import requests

class ObjectDetection:
    # YouTube 동영상에 YOLOv5 구현
    def __init__(self, url, out_file):
        # 객체 생성 시 호출
        # url: 예측 대상 YouTube URL
        # out_file: 유효한 출력 파일 이름 *.avi
        self._URL = url
        self.model = self.load_model()
        self.classes = self.model.names
        self.out_file = out_file
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    def get_video_from_url(self):
        # url에서 새 비디오 스트리밍 객체 생성
        play = pafy.new(self._URL).streams[-1]
        assert play is not None
        return cv2.VideoCapture(play.url)
    def load_model(self):
        # YOLOv5 모델 로드
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model
    def score_frame(self, frame):
        # frame: 단일 프레임; numpy/list/tuple 형식
        # return: 프레임에서 모델이 감지한 객체의 레이블과 좌표
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1].cpu().numpy(), results.xyxyn[0][:, :-1].cpu().numpy()
        return labels
    def class_to_label(self, x):
        # x 숫자 레이블 -> 문자열 레이블로 반환
        return self.classes[int(x)]
    def __call__(self):
        # 인스턴스 생성 시 호출; 프레임 단위로 비디오 로드
        player = self.get_video_from_url()
        assert player.isOpened()
        while True:
            ret, frame = player.read()
            if ret != True:
              break
            if(int(player.get(1)) %40 == 0) :
              results = self.score_frame(frame)
              for i in results:
                self.out_file.append(self.class_to_label(i))

# 파파고 번역
def get_translate(text):
    client_id = "6rBCYe4g5L7a6u38mcn5" # <-- client_id 기입
    client_secret = "qzLdHeUJ0O" # <-- client_secret 기입

    data = {'text' : text,
            'source' : 'en',
            'target': 'ko'}

    url = "https://openapi.naver.com/v1/papago/n2mt"

    header = {"X-Naver-Client-Id":client_id,
              "X-Naver-Client-Secret":client_secret}

    response = requests.post(url, headers=header, data=data)
    rescode = response.status_code

    if(rescode==200):
        send_data = response.json()
        trans_data = (send_data['message']['result']['translatedText'])
        return trans_data
    else:
        print("Error Code:" , rescode)

def Check(url):
  All_list = []
  Video = ObjectDetection(url, All_list)
  Video()
  # 빈도 순으로 정렬
  result = sorted(All_list, key=lambda x: (-All_list.count(x), All_list.index(x)))

  # 중복 제거
  result = reduce(lambda acc, cur: acc if cur in acc else acc+[cur], result, [])
  
  one=result[:11]
  print(one)
  one = get_translate(one)
  if '사람' in one:
    one.remove('사람')
  return one
  #two=one[:5]

  #print(one)
  #one_class, one_valuelist = GetMediaCategory(one)
  #print('모델에 넣은 결과: ',one_class[0],one_valuelist)

  #print(two)
  #two_class, two_valuelist = GetMediaCategory(two)
  #print('모델에 넣은 결과: ',two_class[0], two_valuelist)

Check("https://www.youtube.com/watch?v=c1G23gNlv_8")