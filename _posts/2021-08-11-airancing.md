---
layout: single
title:  "My Ai Racing Game"
font-size: 10px
use_math: true

---
# Machine Learning


# **1. 게임 소개**

* 주어진 서킷을 게이머 차량과 AI 차량이 주행하여 대결하는 게임

# **2. 목표**

* AI 차량의 주행을 구현 
- 로봇이 아닌 사람이 주행하는 것처럼 구현하는 것이 목표
경로나 상황(예를 들어 차량충돌과 같은)에 따른 속도와 경로를 조정하여 부드러운 주행을 구현

# **3. Waypoint를 이용한 경로 설정**

두 종류의 경로 모드가 존재 : SmoothAlongRoute 방식 PointToPoint 방식

* SmoothAlongRoute 방식 : 웨이포인트에 의해 만들어진 부드러운 경로
 - AI가 부드러운 속도와 주행목표지점 조정으로 인간과 자연스러운 주행을 구현

* PointToPoitn 방식 : 오로지 웨이포인트를 순서대로 직선 경로
 - 점대점을 직선으로 이동하므로 AI가 딱딱하고 갑작스러운 주행을 보인다.  


