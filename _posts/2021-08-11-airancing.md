---
layout: single
title:  "My Ai Racing Game"
font-size: 5
use_math: true

---
# Machine Learning

![5](https://user-images.githubusercontent.com/80252681/129064097-d3bba7c7-b5b2-4e56-9049-4cb69ae0698e.jpg)


<br/><br/>
# **1. 게임 소개**

* 서킷을 게이머 차량과 AI 차량이 주행하는 게임

# **2. 설명 내용**

* 주행 경로 만들기
* AI 차량의 주행방식

# **3. Waypoint를 이용한 경로 설정**

두 종류의 경로 모드가 존재

* SmoothRoute 방식 : 웨이포인트를 **부드러운 곡선으로 연결**한 경로<br/>


* NON-SmoothRoute 방식 : 오로지 웨이포인트를 **직선으로 연결**한 경로<br/><br/>

<br/><br/>***간단한 비교***<br/>
<br/><br/>
![3](https://user-images.githubusercontent.com/80252681/129057707-8c70e233-7d89-49f2-937a-6a3f29d8cb93.jpg)<br/><br/>
SmoothRoute 방식 <br/><br/>
![1](https://user-images.githubusercontent.com/80252681/129056815-5201d9db-e12c-41d9-88d2-7f84039c7724.jpg)

NON-SmoothRoute 방식
<br/><br/>
![2](https://user-images.githubusercontent.com/80252681/129058248-c6347365-0dd7-4cbb-a76c-f18c47c4db78.jpg)


<br/><br/>
## **3.1 SmoothRoute 방식 설명**
<br/>

* 핵심은 웨이포인트들을 **부드러운 곡선으로 연결하는 방식**이다.<br/>
(NON-SmoothRoute 방식은 직선으로 연결하는 것이니 추가 설명은 않는다.)
<br/>

<br/><br/>
웨이포인트를 연결하여 경로를 얻는 부분입니다.
<br/><br/>

```csharp
if (smoothRoute)
            {
                // smooth catmull-rom calculation between the two relevant points


                // get indices for the surrounding 2 points, because
                // four points are required by the catmull-rom function
                p0n = ((point - 2) + numPoints)%numPoints;
                p3n = (point + 1)%numPoints;

                // 2nd point may have been the 'last' point - a dupe of the first,
                // (to give a value of max track distance instead of zero)
                // but now it must be wrapped back to zero if that was the case.
                p2n = p2n%numPoints;

                P0 = points[p0n];
                P1 = points[p1n];
                P2 = points[p2n];
                P3 = points[p3n];

                return CatmullRom(P0, P1, P2, P3, i);
            }

```
<br/>

여기서 주목해야 할 부분은 최종적으로 리턴하는 **CatmullRom** 입니다.<br/>

```csharp
private Vector3 CatmullRom(Vector3 p0, Vector3 p1, Vector3 p2, Vector3 p3, float i)
        {
            // comments are no use here... it's the catmull-rom equation.
            // Un-magic this, lord vector!
            return 0.5f*
                   ((2*p1) + (-p0 + p2)*i + (2*p0 - 5*p1 + 4*p2 - p3)*i*i +
                    (-p0 + 3*p1 - 3*p2 + p3)*i*i*i);
        }
```
CatmullRom은 **캣멀롬** 스플라인 이라고 하는데 네 점(p0~p4)이 주어지면 두 점을 연결하는 부드러운 곡선이 정의되는 공식입니다.<br/>

![4](https://user-images.githubusercontent.com/80252681/129063607-7b830670-c8b5-4228-aa5a-a8fc0415cbb3.png)

컴퓨터 그래픽에서 한 점에서 한점으로 이동하는 대상의 부드러운 움직임을 구현할 때 많이 사용됩니다.<br/>
~~부드러운 곡선을 만드는 것으로 **베지어 곡선**이란 것도 있는데, 둘의 큰 차이점은 캣멀롬은 웨이포인트를 통과하고 베지어는 꼭 통과하지 않고 유사하게 움지역 곡선을 만든다는 차이가 있습니다~~
