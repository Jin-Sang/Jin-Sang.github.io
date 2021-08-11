---
layout: single
title:  "My Ai Racing Game"
font-size: 5
use_math: true

---
# Machine Learning
<br/>

![5](https://user-images.githubusercontent.com/80252681/129064097-d3bba7c7-b5b2-4e56-9049-4cb69ae0698e.jpg)


<br/><br/>
# **1. 게임 소개** &#128661;

* 서킷을 게이머 차량과 AI 차량이 주행하는 게임<br/><br/>

# **2. 설명 내용** &#128655;

* 주행 경로 만들기
* AI 차량의 서킷운행 방식
* AI 차량의 주행 방식<br/><br/>

# **3. Waypoint를 이용한 경로 설정** &#128678;

(참고 스크립트 : WaypointCircuit.cs)

두 종류의 경로 모드가 존재

* **SmoothRoute** 방식 : 웨이포인트를 **부드러운 곡선으로 연결**한 경로<br/>


* **NON-SmoothRoute** 방식 : 오로지 웨이포인트를 **직선으로 연결**한 경로<br/><br/>

<br/><br/>***간단한 비교***<br/>
<br/><br/>
![3](https://user-images.githubusercontent.com/80252681/129057707-8c70e233-7d89-49f2-937a-6a3f29d8cb93.jpg)<br/><br/>
SmoothRoute 방식 <br/><br/>
![1](https://user-images.githubusercontent.com/80252681/129056815-5201d9db-e12c-41d9-88d2-7f84039c7724.jpg)<br/>
부드러운 곡선 모양의 경로

NON-SmoothRoute 방식
<br/><br/>
![2](https://user-images.githubusercontent.com/80252681/129058248-c6347365-0dd7-4cbb-a76c-f18c47c4db78.jpg)<br/>
직선 모양의 경로


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

주목해야 할 부분은 최종적으로 리턴하는 **CatmullRom** 입니다.<br/>

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

<br/><br/>***IN-GAME 경로 생성 사진***<br/><br/>

SmoothRoute 방식 <br/><br/>

![6](https://user-images.githubusercontent.com/80252681/129066023-fa6f2f1f-4cbc-4e36-83bf-a56a54ed2104.jpg)

부드러운 곡선의 경로가 나타난 것이 확인됩니다.

NON-SmoothRoute 방식
<br/><br/>
![7](https://user-images.githubusercontent.com/80252681/129066358-dbe6e327-4178-4672-b7bf-4e612fa608c2.jpg)

각이 부드럽지 않고 뾰족한 경로가 생성되었습니다.<br/><br/>



경로생성|특징요약
---|---
SmoothRoute|곡선 경로
Non-SmoothRoute|직선 경로




<br/><br/><br/>
이제 경로를 만들었으니<br/>
AI차량이 어떻게 주행하는지 보겠습니다.
<br/><br/>


# **4. AI 차량의 서킷운행 모드**&#128665;

(참고 스크립트 : WaypointProgressTracker.cs)

두 종류의 서킷운행 모드가 존재<br/><br/>

* **SmoothAlongRoute** 모드 : 생성된 경로를 인식하여 **현재 위치에서 경로의 가까운 앞을 목표**로 삼아 운행<br/><br/>
( 실제로 사람이 눈으로 가까운 앞을 보고 운전하듯이 운행한다. ) <br/>

<br/> ***차량 앞의 초록색(형광색) 선은 차량이 향하고자 하는 목표입니다.***

{% include video id="gyy6s5fxehE" provider="youtube" %}

차량이 **주행함에 따라 경로를 따라 목표가 바뀌고 있음**이 확인됩니다. (마치 앞을 보고 운전하는 사람같이)<br/><br/>
```csharp
        [SerializeField] private float lookAheadForTargetOffset = 5;
        // The offset ahead along the route that the we will aim for

        [SerializeField] private float lookAheadForTargetFactor = .1f;
        // A multiplier adding distance ahead along the route to aim for, based on current speed

        [SerializeField] private float lookAheadForSpeedOffset = 10;
        // The offset ahead only the route for speed adjustments (applied as the rotation of the waypoint target transform)

        [SerializeField] private float lookAheadForSpeedFactor = .2f;
        // A multiplier adding distance ahead along the route for speed adjustments

        [SerializeField] private ProgressStyle progressStyle = ProgressStyle.SmoothAlongRoute;
        // whether to update the position smoothly along the route (good for curved paths) or just when we reach each waypoint.
```

SmoothAlongRoute 모드에서 **현위치에서 앞을 보는 범위(목표설정범위)를 설정**하는 범위(Offset)와 가중치(Factor)입니다.<br/>
**Target(순간순간의 운행목표)와 Speed(속도) 조절을 위한 미리보기** 범위(목표설정범위)를 설정하는 것입니다.<br/>
즉, **항상 경로의 현위치에서 일정량 앞**을 보고 있습니다. <br/><br/>
밑의 코드를 보면 이해가 더 잘 됩니다.<br/>

```csharp
if (progressStyle == ProgressStyle.SmoothAlongRoute)
            {
                // determine the position we should currently be aiming for
                // (this is different to the current progress position, it is a a certain amount ahead along the route)
                // we use lerp as a simple way of smoothing out the speed over time.
                if (Time.deltaTime > 0)
                {
                    speed = Mathf.Lerp(speed, (lastPosition - transform.position).magnitude/Time.deltaTime,
                                       Time.deltaTime);
                }
                target.position =
                    circuit.GetRoutePoint(progressDistance + lookAheadForTargetOffset + lookAheadForTargetFactor*speed)
                           .position;
                target.rotation =
                    Quaternion.LookRotation(
                        circuit.GetRoutePoint(progressDistance + lookAheadForSpeedOffset + lookAheadForSpeedFactor*speed)
                               .direction);


```

progressDistance(경로에서 현재 거리)에서 Offset(미리보는 범위)와 Factor(가중치)에 속도를 곱한 값을 더해서 그 만큼 앞을 보고있습니다.<br/>
즉, **Offset(미리보는 범위)와 Factor(가중치)에 속도를 곱한 값 만큼 앞**을 목표로 보고 있습니다.<br/><br/><br/>

**&#128051;중요 : 어떻게 현위치를 파악하고 있을까?**<br/><br/>

* progressDistance(경로상 현재 진행 거리)를 어떻게 알 수 있을까?<br/>
차량은 운행하다보면 경로를 벗어날 수 있는데 그럴 땐 ***어떻게 벗어난 지점으로 돌아오지 않고 자연스럽게 경로 진행 방향 이동하며 다시 돌아오는 것***인지 궁금했습니다.<br/><br/>

현 위치를 파악하는 코드
```csharp
// get our current progress along the route
                progressPoint = circuit.GetRoutePoint(progressDistance);
                Vector3 progressDelta = progressPoint.position - transform.position;
                if (Vector3.Dot(progressDelta, progressPoint.direction) < 0)
                {
                    progressDistance += progressDelta.magnitude*0.5f;
                }

```
코드는 짧았지만, 해석하는데 오랜 시간이 걸렸습니다.
```csharp
Vector3.Dot(progressDelta, progressPoint.direction) < 0
```
특히 이 부분에서 오랜 시간이 걸렸습니다.<br/>
먼저 progressPoint는 현재 경로 상으로 따졌을 때 진행위치입니다. progressDistance 경로를 따라 진행된 거리에 의해 구해집니다.<br/>
차량은 항상 경로위에만 있을 수 없습니다.(물리적인 이유)<br/>
***progressDistance는 단순히 얼마나 움직인 거리를 말하는 것이 아니라 경로를 기준으로 얼마만큼 경로의 진행방향으로 진행했냐는 것입니다.***<br/><br/>

progressDelta는 progressPoint(지금까지 운행을 경로 위에 표시했을 때 위치, 즉 진행 상태)와 실제 지금 차량 위치의 차이를 나타내는 벡터입니다. 


![8](https://user-images.githubusercontent.com/80252681/129087184-91690d1c-90ad-4fcf-a6c6-edc2662041d2.jpg)
![9](https://user-images.githubusercontent.com/80252681/129085101-9aef832c-344f-463a-a371-0ffec76cba64.jpg)<br/>
즉, 보라색 벡터가 progressDelta 입니다.


```csharp
Vector3.Dot
```
이것은 내적을 말합니다. <br/><br/>
![10](https://user-images.githubusercontent.com/80252681/129086079-7e8cda7c-e4d7-4afa-9c66-dc103de4a35a.jpg)<br/>
$\theta$ 는 두 벡터 사이의 각을 말합니다.<br/><br/><br/>

그렇다면 이제 다시 한 번 보겠습니다.
```csharp
Vector3.Dot(progressDelta, progressPoint.direction) < 0
```
먼저 어떤 벡터들의 내적을 묻는지 파악해야 합니다.<br/>
progressPoint.direction는 경로 위로 봤을 때 진행상황에서 위치의 방향을 말합니다.<br/>
![11](https://user-images.githubusercontent.com/80252681/129087813-09f579b7-b50e-4c2d-93c6-040299ffb980.jpg)<br/>
즉 주황색 벡터를 말하며 결론적으로 주황색 벡터와 보라색 벡터의 내적을 사용합니다.<br/>
![12](https://user-images.githubusercontent.com/80252681/129088334-5cf0d8da-83bd-4696-a86f-d43567d86786.jpg)<br/><br/>

그렇다면 이 내적값이 음수일 때 
```csharp
if (Vector3.Dot(progressDelta, progressPoint.direction) < 0)
                {
                    progressDistance += progressDelta.magnitude*0.5f;
                }
```
진행 거리를 그 현 위치와 진행상 위치의 차이만큼 더해주라고 합니다.<br/>
즉, 쉽게 말해 내적값이 음수일 때 진행이 된다는 것입니다.<br/><br/>

**&#128064;내적값이 음수가 의미하는 것은?**

내적값이 음수 이려면 $\cos\theta$ 가 음수여야 하므로,<br/>
$\theta$ 가 $90^\circ$보다 크고 $180^\circ$ 보다 작아야 한다. <br/>
![13](https://user-images.githubusercontent.com/80252681/129091558-9b103ed9-bdf2-4e25-a638-7efb060c6b1b.jpg)<br/>
차량의 3 위치 중 어떨 때 해당이 될까? <br/>
당연히 차량이 경로상 진행 위치보다 경로의 진행 방향을 따라 더 앞으로 운행했을 경우이다.<br/>
즉, 경로 상 그 위치를 통과한 경우(진행 방향상 더 앞으로 이동한 경우) 그 진행 거리를 인정해주어 더해준다는 것이다.<br/>
이 코드가 이동한 거리만큼 
그만큼 경로 진행을 했다는 것이고, 그래서 미리보기(목표)가 경로 진행을 따라 같이 진행할 수 있는 것이다.<br/><br/>

***쉽게 말해 각을 통해 경로 상 위치와 $90^\circ$를 이루는 위치까지 인정해준다는 것이다.***<br/>
![15](https://user-images.githubusercontent.com/80252681/129093917-03741eb8-c9d8-45d6-86a9-d53eba010d3a.jpg)<br/><br/>

![16](https://user-images.githubusercontent.com/80252681/129095367-2435b40c-0087-4d31-bb82-907a91fd021b.jpg)<br/>
그래서 경로를 이탈한 경우 이탈한 지점으로 돌아오지 않고 경로상 위치를 인정받고 <br/>
인정받은 위치를 기준으로 약간의 앞을 목표로 합니다.











* **PointToPoint** 모드 : 생성된 **경로를 인식하지 않고 웨이포인트를 순차적으로 목표**로 삼아 운행<br/><br/>
( 경로를 무시하고 웨이포인트만 보고 간다.)<br/><br/>

{% include video id="pTRe75TJ1O0" provider="youtube" %}

차량이 **경로와 현위치를 무시하고 오직 웨이포인트만을 목표**로 설정하고 있습니다.<br/><br/>

```csharp
 // point to point mode. Just increase the waypoint if we're close enough:

                Vector3 targetDelta = target.position - transform.position;
                if (targetDelta.magnitude < pointToPointThreshold)
                {
                    progressNum = (progressNum + 1)%circuit.Waypoints.Length;
                }


                target.position = circuit.Waypoints[progressNum].position;
                target.rotation = circuit.Waypoints[progressNum].rotation;
```
차량이 웨이포인트의 일정거리(pointToPointThreshold)안으로 들어오면 다음 순서의 웨이포인트를 목표로 바꾸고 운행한다.<br/><br/><br/>

***실제 주행 중인 차량이 목표를 어디로 보면서 가느냐의 문제인데 이것은 AI차량의 주행(5장) 부분에서 언급하겠지만 속도 조절과 관련되어 부드러운 주행에 큰 영향을 끼친다. 실제 인간은 차의 약간 앞을 보면서(목표)로 하고 주행한다.***<br/><br/><br/>



운행방식|특징요약
---|---
SmoothAlongRoute|운행 by 경로
PointToPoint|운행 by 웨이포인트

<br/><br/><br/><br/>


# **5. AI 차량의 주행 방식**&#128692;
(참고 스크립트 : CarAiControl.cs)
```csharp
case BrakeCondition.TargetDirectionDifference:
                        {
                            // the car will brake according to the upcoming change in direction of the target. Useful for route-based AI, slowing for corners.

                            // check out the angle of our target compared to the current direction of the car
                            float approachingCornerAngle = Vector3.Angle(m_Target.forward, fwd);

                            // also consider the current amount we're turning, multiplied up and then compared in the same way as an upcoming corner angle
                            float spinningAngle = m_Rigidbody.angularVelocity.magnitude*m_CautiousAngularVelocityFactor;

                            // if it's different to our current angle, we need to be cautious (i.e. slow down) a certain amount
                            float cautiousnessRequired = Mathf.InverseLerp(0, m_CautiousMaxAngle,
                                                                           Mathf.Max(spinningAngle,
                                                                                     approachingCornerAngle));
                            desiredSpeed = Mathf.Lerp(m_CarController.MaxSpeed, m_CarController.MaxSpeed*m_CautiousSpeedFactor,
                                                      cautiousnessRequired);
                            break;
```
구체적으로 AI 차량이 속도를 조절하는 부분입니다.<br/>
4장에서 설명한 **타깃(목표) 설정에 따라 즉, 미리보기(타깃)를 파악**하여 AI는 목표와의 각도와 현재 회전 중인 양을 고려하여 속도를 조절합니다.<br/><br/>

따라서 4장에서의 목표 설정이 부드러운 주행에 중요한 이유가 여기서 드러납니다.<br/>
* 지속적으로 부드러운 경로에 맞게 현재 주행에 따라 목표를 설정해주는 **SmoothAlongRoute**의 경우,
**연속적이고 부드러운 목표 설정이 계속 이루어지기에** 주행 또한 자연스럽습니다.<br/><br/>

* 반면, **PointToPoint의 경우**는 **목표 설정이 비교적 시간적, 공간적으로 간격이 큽니다.**
따라서 갑작스러운 목표 변화에 따른 갑작스러운 속도 변화 또는 방향 전환 등, 
부자연스러운 주행이 될 수 밖에 없습니다.<br/><br/><br/><br/>


***최종 정리***<br/><br/>

먼저 3장에서  &#128678;경로 만드는 법을 보았고,<br/>
4장에서 &#128665;운행 중 목표를 설정하는 법을 보았고,<br/>
마지막으로 5장에서 &#128692;AI가 차량을 운행 하는 방식을 살펴보았습니다.

인간이 목적지의 경로를 파악하고(3장), 순간순간 목표를 확인하고(4장), 실제 차량을 모는 것(5장)<br/>
과 같은 순서로 알아보았습니다. 
