---
layout: single
title:  "My Ai Racing Game"
font-size: 5
use_math: true

---
# Machine Learning

<br/><br/>
# **1. 게임 소개**

* 서킷을 게이머 차량과 AI 차량이 주행하는 게임

# **2. 목표**

* AI 차량의 주행을 구현 

# **3. Waypoint를 이용한 경로 설정**

두 종류의 경로 모드가 존재

* SmoothAlongRoute 방식 : 웨이포인트를 **부드럽게 연결** 경로<br/><br/>
AI가 부드러운 속도와 주행목표지점 조정으로 인간과 자연스러운 주행을 구현<br/>


* PointToPoitn 방식 : 오로지 웨이포인트를 **직선으로 연결**한 경로<br/><br/>
점대점을 직선으로 이동하므로 AI가 딱딱하고 갑작스러운 주행을 보인다.

<br/><br/>***간단한 비교***<br/>
<br/><br/>
![3](https://user-images.githubusercontent.com/80252681/129057707-8c70e233-7d89-49f2-937a-6a3f29d8cb93.jpg)<br/><br/>
SmoothAlongRoute 방식 <br/><br/>
![1](https://user-images.githubusercontent.com/80252681/129056815-5201d9db-e12c-41d9-88d2-7f84039c7724.jpg)

PointToPoint 방식
<br/><br/>
![2](https://user-images.githubusercontent.com/80252681/129058248-c6347365-0dd7-4cbb-a76c-f18c47c4db78.jpg)



## **3.1 SmoothAlongRoute 방식 설명**
<br/>
* 핵심은 웨이포인트들을 부드러운 곡선으로 연결하는 방식이다.
  * PointToPoint 방식은 직선으로 연결하는 것이니 추가 설명은 않는다.


