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

# 🏭 공장 구조 인식 시뮬레이터 (Factory Layout Detector)

이 Gradio 앱은 업로드된 공장 작업 영상을 분석하여, 영상 내 구조물(사람, 지게차, 컨베이어 벨트, 기계)의 위치를 감지하고 이를 기반으로 2D 공장 미니맵을 생성합니다.

## 🚀 기능
- 사람, 지게차, 기계, 컨베이어 벨트 감지 (YOLOv8 기반)
- 영상 프레임을 분석하여 객체의 평균 위치 추론
- 2D 평면도 시각화 출력

## 🧠 모델
- `YOLOv8m.pt` 사용 (Ultralytics, COCO pretrained)
- 향후 구조물 감지를 위한 커스텀 모델로 확장 가능

## 📦 사용법
1. 상단의 '공장 작업 영상 업로드'에 .mp4 또는 .avi 파일 업로드
2. 분석 완료 후, 자동으로 2D 미니맵 생성됨

## 📌 향후 계획
- 커스텀 데이터셋 기반 fine-tuning
- 미니맵 내 시간 흐름에 따른 동선 애니메이션



Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
