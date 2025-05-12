---
title: Factory Layout Simulator
emoji: 😻
colorFrom: red
colorTo: pink
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: false
license: mit
short_description: 비전 기반으로 환경을 분석하고 시각화하는 공장 작업장 구조 시뮬레이터
---

# 🏭 YOLO 8 기반 공장 구조 인식 시뮬레이터 (Factory Layout Detector)
1. 업로드된 공장 작업 영상을 분석하여, 영상 내 구조물(사람, 지게차, 컨베이어 벨트, 기계)의 위치를 감지하고 이를 기반으로 2D 공장 미니맵을 생성.
2. https://huggingface.co/spaces/shin0624/Factory_Layout_Simulator

![Image](https://github.com/user-attachments/assets/1011085a-b214-4a4c-af2d-3d9f47ebf4f8)

## 사용 예시
![Image](https://github.com/user-attachments/assets/a76cd7be-b52e-4439-845b-237633bcb652)

## 기술 스택
![License](https://img.shields.io/badge/License-MIT%2FAGPL--3.0-blue)
<img src="https://img.shields.io/badge/huggingface-FFD21E?style=for-the-badge&logo=huggingface&logoColor=white">
<img src="https://img.shields.io/badge/yolo11-111F68?style=for-the-badge&logo=yolo&logoColor=white">
<img src="https://img.shields.io/badge/Gradio-F97316?style=for-the-badge&logo=Gradio&logoColor=white">
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-%23F9A825.svg?style=for-the-badge&logo=googlecolab&logoColor=white)

## 🚀 기능
- 사람, 지게차, 기계, 컨베이어 벨트 감지 (YOLOv8 기반)
- 영상 프레임을 분석하여 객체의 평균 위치 추론
- 2D 평면도 시각화 출력

## 🧠 모델
- `YOLOv8m.pt` 사용 (Ultralytics, COCO pretrained)
- 향후 구조물 감지를 위한 커스텀 모델로 확장 가능

## 📦 사용법
1. 상단의 '공장 작업 영상 업로드'에 .mp4 또는 .avi 파일 업로드(다수의 파일 업로드 가능)
2. 분석 완료 후, 자동으로 2D 미니맵 생성됨

## 📌 향후 계획
- 커스텀 데이터셋 기반 fine-tuning
- 미니맵 내 시간 흐름에 따른 동선 애니메이션



Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
