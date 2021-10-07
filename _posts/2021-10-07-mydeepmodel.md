
# **저수준 선형 분류 신경망 추가기능 구현**
---
추가적인 기능을 구현한다.
* 미니 배치 학습 (성공)
* 정확도 측정 (성공)
* 3개의 층 구현 (성공)

결과적으로 모든 도전 과제를 성공적으로 추가하였습니다.


```python
import tensorflow as tf
import numpy as np
```

텐서플로우와 Numpy를 임포트 합니다.


## **데이터셋 준비** 

- `np.random.multivariate_normal()`
    - 다변량 정규분포를 따르는 데이터 생성 (특성이 2개)
    - 평균값과 공분산 지정 필요 (타원 모양의 데이터 퍼진 모양)
- 음성 데이터셋
    - 샘플 수: 1,000
    - 평균값: `[0, 3]`
    - 공분산: `[[1, 0.5],[0.5, 1]]`
- 양성 데이터셋
    - 샘플 수: 1,000
    - 평균값: `[3, 0]`
    - 공분산: `[[1, 0.5],[0.5, 1]]`


```python
num_samples_per_class = 1000

# 음성 데이터셋
negative_samples = np.random.multivariate_normal(
    mean=[0, 3], cov=[[1, 0.5],[0.5, 1]], size=num_samples_per_class)

# 양성 데이터셋
positive_samples = np.random.multivariate_normal(
    mean=[3, 0], cov=[[1, 0.5],[0.5, 1]], size=num_samples_per_class)
```

양성과 음성의 각각의 1000개씩의 데이터를 합쳐서 `(2000, 2)` 모양의 데이터셋을 만들어 줍니다. 그리고 자료형을 `np.float32`로 지정해줍니다. 메모리 사용공간과 실행시간을 줄이기 위함입니다.


```python
inputs = np.vstack((negative_samples, positive_samples)).astype(np.float32)
```

데이터셋에 해당하는 타깃(레이블)도 만들어줍니다. 음성은 `0`, 양성은 `1`로 설정해줍니다.


```python
targets = np.vstack((np.zeros((num_samples_per_class, 1), dtype="float32"),
                     np.ones((num_samples_per_class, 1), dtype="float32")))
```

양성, 음성 샘플을 좌표에서 색깔로 구분하여 확인해보겠습니다. 

- `inputs[:, 0]`: x 좌표
- `inputs[:, 1]`: y 좌표
- `c=targets[:, 0]`: `0` 또는 `1` (레이블)에 따른 색상 지정




```python
import matplotlib.pyplot as plt

plt.scatter(inputs[:, 0], inputs[:, 1], c=targets[:, 0])
plt.show()
```


![png](output_11_0.png)


## **가중치 변수 텐서 생성(3층구현)**


```python
inter_layers_dim1 = 2  # 1층에서 2층으로 갈 때 특성 수 
inter_layers_dim2 = 2   # 2층에서 3층으로 갈 때 특성 수 
```

층 간의 연결을 위해 넘겨주는 특성 수를 설정해주었습니다.

이 값들을 조절하면서 모델의 성능을 비교해가며 더 좋은 모델을 찾기 위해 노력할 수 있습니다. 

* **1층 덴스층(은닉층) 가중치와 편향**


```python
input_dim1 = 2                     # 입력 샘플의 특성수
output_dim1 = inter_layers_dim1    # 출력 샘플의 특성수

# 가중치: 무작위 초기화
W1 = tf.Variable(initial_value=tf.random.uniform(shape=(input_dim1, output_dim1)))

# 편향: 0으로 초기화
b1 = tf.Variable(initial_value=tf.zeros(shape=(output_dim1,)))
```

입력 데이터의 특성수는 2개. 1층 출력 특성수는 위에서 설정한 5개입니다. 그리고 가중치와 편향을 초기화 해줍니다.

* **2층 덴스층(은닉층) 가중치와 편향**


```python
input_dim2 = inter_layers_dim1     # 입력 샘플의 특성수
output_dim2 = inter_layers_dim2    # 하나의 값으로 출력

# 가중치: 무작위 초기화
W2 = tf.Variable(initial_value=tf.random.uniform(shape=(input_dim2, output_dim2)))

# 편향: 0으로 초기화
b2 = tf.Variable(initial_value=tf.zeros(shape=(output_dim2,)))
```

2층에서는 1층의 출력값을 받기 때문에 1층 출력값의 특성 갯수를 입력 데이터 특성수 즉, 5 개로 받고 출력도 5개로 해주는 가중치와 편향을 초기화해줍니다.

* **3층 덴스층(출력층) 가중치와 편향**


```python
input_dim3 = inter_layers_dim2     # 입력 샘플의 특성수
output_dim3 = 1                    # 하나의 값으로 출력

# 가중치: 무작위 초기화
W3 = tf.Variable(initial_value=tf.random.uniform(shape=(input_dim3, output_dim3)))

# 편향: 0으로 초기화
b3 = tf.Variable(initial_value=tf.zeros(shape=(output_dim3,)))
```

마지막 출력 층입니다. 2층에서의 출력을 받아서 1개의 특성의 값을 가지게 하는 가중치와 편향을 초기화해줍니다.

## **예측 모델(함수) 선언**
---
아래 함수는 각각의 1개 층에서 모델의 출력값을 계산하는 것을 정의합니다.

* **1층 덴스층(은닉층) 함수**


```python
def layer1(inputs, activation=None):
    outputs = tf.matmul(inputs, W1) + b1
    if activation != None:
        return activation(outputs)
    else:
        return outputs
```

입력 데이터를 받아서 1층 가중치와 행렬곱셈을 해주고, 편향을 더해서(아핀변환) 출력값을 만듭니다.

* **2층 덴스층(은닉층) 함수**


```python
def layer2(inputs, activation=None):
    outputs = tf.matmul(inputs, W2) + b2
    if activation != None:
        return activation(outputs)
    else:
        return outputs
```

1층에서 받은 데이터를 다시 2층 가중치와 행렬곱셈을 해주고, 편향을 더해서(아핀변환)  출력값을 만듭니다.

* **3층 덴스층(출력층) 함수**


```python
def layer3(inputs, activation=None):
    outputs = tf.matmul(inputs, W3) + b3
    if activation != None:
        return activation(outputs)
    else:
        return outputs
```

2층에서 받은 데이터를 다시 3층 가중치와 행렬곱셈을 해주고, 편향을 더해서(아핀변환) 최종 1개의 특성값을 가지는 출력값을 만듭니다.


```python
def model(inputs):
    layer1_outputs = layer1(inputs, tf.nn.relu)
    layer2_outputs = layer2(layer1_outputs, tf.nn.relu)
    layer3_outputs = layer3(layer2_outputs)
    return layer3_outputs
```

마지막으로 `model`함수는 각층을 순서대로 실행하고(활성화함수도 동작함) 최종 출력값을 반환하게 합니다. 

## **손실함수 : 평균 제곱 오차(MSE)**
---
`tf.reduce_mean()`: 텐서에 포함된 항목들의 평균값 계산.
    넘파이의 `np.mean()`과 결과는 동일하지만 텐서플로우의 텐서를 대상으로 함.


```python
def square_loss(targets, predictions):
    per_sample_losses = tf.square(targets - predictions)
    return tf.reduce_mean(per_sample_losses)
```

타깃과 실제 예측의 차를 구해서 제곱한 것들의 평균을 내는 것입니다.

## **정확도 함수 : accuracy**
---
인간이 성능을 쉽게 파악하기 위한 수치이다.


```python
def my_accuracy(inputs, targets):
  predictions = model(inputs) # 훈련된 모델을 통해 예측을 합니다.
  predictions = np.where(predictions > 0.5, 1.0, predictions) # 0.5보다 큰 값이 나오면 1로 바꿔줍니다.
  predictions = np.where(predictions <= 0.5, 0.0, predictions) # 0.5보다 작은 값이 나오면 0으로 바꿔줍니다.

  accuracy = np.mean(np.equal(predictions,targets)) # 레이블과 비교하여 정확도를 구합니다.
  return accuracy

```

훈련된 모델에 데이터를 넣어서 예측값을 만듭니다. 결국 지금 만들어내야하는 결과는 음성이냐 양성이냐 둘 중 하나를 선택하는 것이므로 0.5보다 클때는 1(양성)로 작을 때는 0(음성)으로 예측하도록 하였습니다 마지막으로 레이블과 비교하여 얼마나 일치하는지를 계산하여 정확도를 구하였습니다.

## **훈련**

정해진 한번의 배치를 예측 후 `GradientTape()`을 이용하여 그레이디언트를 계산해 각각의 층의 가중치와 편향을 한 번 조정하는 함수 입니다. 즉, 경사하강법을 구현하고 있습니다.


```python
learning_rate = 0.1

def training_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = square_loss(predictions, targets)
    grad_loss_wrt_W1, grad_loss_wrt_b1, grad_loss_wrt_W2, grad_loss_wrt_b2, grad_loss_wrt_W3, grad_loss_wrt_b3 = tape.gradient(loss, [W1, b1, W2, b2, W3, b3])
    W1.assign_sub(grad_loss_wrt_W1 * learning_rate)
    b1.assign_sub(grad_loss_wrt_b1 * learning_rate)
    W2.assign_sub(grad_loss_wrt_W2 * learning_rate)
    b2.assign_sub(grad_loss_wrt_b2 * learning_rate)
    W3.assign_sub(grad_loss_wrt_W3 * learning_rate)
    b3.assign_sub(grad_loss_wrt_b3 * learning_rate)
    return loss
```

학습률에 따라 각각의 가중치와 편향을 경사하강법으로 학습합니다.

## **미니배치 훈련**

* **미니 배치 구현하기**

해당함수는 한번의 가중치 조절을 하므로 훈련세트를 미니 배치로 나눠서 순서대로 `traning_step()`함수를 호출하면 될 것입니다.  

### **미니 배치 만들기**
---
데이터셋을 미니 배치(100개)로 나눕니다.
* 2000개의 샘플이 있으므로 100개씩 나누려면 20개로 나누어야합니다.


```python
mini_batch = np.vsplit(inputs, 20)

mini_batch[:3]
```




    [array([[-8.15438569e-01,  7.98837721e-01],
            [ 1.34129834e+00,  3.45491838e+00],
            [ 9.02706623e-01,  2.90800643e+00],
            [ 1.06513572e+00,  3.92177057e+00],
            [-1.37700453e-01,  2.33954906e+00],
            [-5.09217799e-01,  1.11027229e+00],
            [-1.62421063e-01,  2.87157321e+00],
            [ 9.06001329e-01,  3.23060012e+00],
            [ 7.27615952e-01,  2.80088472e+00],
            [ 1.68395185e+00,  5.91812038e+00],
            [ 9.27531719e-01,  3.13468575e+00],
            [-5.25513589e-01,  2.21611428e+00],
            [ 1.79990685e+00,  2.63261175e+00],
            [-2.78466016e-01,  2.53581548e+00],
            [-4.30744231e-01,  3.84137106e+00],
            [ 1.32226273e-01,  2.99246669e+00],
            [ 5.82473934e-01,  3.75747275e+00],
            [-2.08569384e+00,  1.77356100e+00],
            [-2.78174907e-01,  2.98395848e+00],
            [-8.83099362e-02,  3.32570624e+00],
            [ 4.05880451e-01,  2.85628724e+00],
            [ 1.64813554e+00,  4.01705456e+00],
            [-5.64625800e-01,  2.50329494e+00],
            [ 9.16824341e-01,  3.58431077e+00],
            [-3.50827388e-02,  3.65199447e+00],
            [-3.89760472e-02,  3.56220675e+00],
            [ 4.71734345e-01,  4.53853416e+00],
            [-8.17948103e-01,  2.17931628e+00],
            [ 5.02125561e-01,  4.44144535e+00],
            [-1.26563776e+00,  6.03283405e-01],
            [-7.93355823e-01,  3.64675570e+00],
            [ 1.19693220e+00,  1.87920964e+00],
            [-1.28584802e-01,  2.71022129e+00],
            [-2.14207605e-01,  1.79547107e+00],
            [ 6.71810329e-01,  2.91438603e+00],
            [ 6.09785795e-01,  4.45633554e+00],
            [ 1.68155456e+00,  2.88619399e+00],
            [ 2.78366029e-01,  4.68857288e+00],
            [ 8.04914534e-01,  3.86958122e+00],
            [-9.56894457e-02,  3.30337119e+00],
            [ 1.59036887e+00,  3.71655560e+00],
            [-2.82287657e-01,  3.34349227e+00],
            [ 9.25207287e-02,  3.93135118e+00],
            [ 1.30264461e-01,  2.53301954e+00],
            [-7.92860210e-01,  6.09980464e-01],
            [-7.56978989e-01,  2.67316818e+00],
            [ 2.29369938e-01,  4.32808304e+00],
            [-5.06989181e-01,  2.51676178e+00],
            [-1.66488826e-01,  4.46441603e+00],
            [ 8.11000347e-01,  1.75146413e+00],
            [-1.82646692e-01,  3.59662342e+00],
            [ 7.15064704e-01,  4.38288021e+00],
            [ 1.63291663e-01,  3.07949209e+00],
            [ 2.07943201e+00,  3.26039553e+00],
            [-1.58922637e+00,  1.62242281e+00],
            [-2.41414905e-01,  4.66812229e+00],
            [-2.15940744e-01,  2.83275962e+00],
            [-2.82216263e+00,  1.30095351e+00],
            [-5.51692426e-01,  1.58225274e+00],
            [-1.11391473e+00,  1.48876143e+00],
            [ 1.23545587e+00,  4.16570187e+00],
            [ 7.92077720e-01,  2.22428012e+00],
            [-1.24491006e-01,  3.16683173e+00],
            [ 5.24503171e-01,  3.10941720e+00],
            [ 3.83248061e-01,  2.86110330e+00],
            [-1.27898142e-01,  4.42980051e+00],
            [-1.02036428e+00,  2.20207810e+00],
            [-1.43822503e+00,  3.11734176e+00],
            [-1.88770428e-01,  3.63220382e+00],
            [ 6.04223728e-01,  3.16505551e+00],
            [ 9.58272398e-01,  4.62947607e+00],
            [-5.74088335e-01,  4.77997351e+00],
            [ 8.01129878e-01,  3.22812009e+00],
            [ 3.68785323e-03,  3.61726570e+00],
            [-6.08945787e-01,  3.24321413e+00],
            [ 1.73390758e+00,  3.95787549e+00],
            [ 9.38997447e-01,  2.87740088e+00],
            [-8.11442614e-01,  3.33624792e+00],
            [ 6.61258459e-01,  3.44471025e+00],
            [ 6.53733194e-01,  3.61788201e+00],
            [-4.11357939e-01,  2.75878787e+00],
            [ 2.47724605e+00,  4.44584990e+00],
            [ 9.67495203e-01,  3.16796613e+00],
            [-3.82482827e-01,  3.44381928e+00],
            [ 1.78254560e-01,  1.72960246e+00],
            [ 1.09487903e+00,  3.41319656e+00],
            [ 1.50140297e+00,  3.12543130e+00],
            [ 7.95699120e-01,  4.55069923e+00],
            [-1.14595139e+00,  6.27900660e-01],
            [-1.27008224e+00,  2.81473589e+00],
            [ 1.95343331e-01,  3.21792126e+00],
            [ 2.58162379e-01,  2.72458887e+00],
            [-7.00233340e-01,  3.07330966e+00],
            [ 5.46936095e-02,  2.90963602e+00],
            [ 1.58273607e-01,  3.16504741e+00],
            [-4.81177539e-01,  4.52548265e+00],
            [-6.37600660e-01,  3.41156173e+00],
            [ 5.71779311e-01,  2.68273544e+00],
            [-1.36386678e-02,  2.52556658e+00],
            [ 1.06162941e+00,  1.12285376e+00]], dtype=float32),
     array([[-0.5492963 ,  1.7587523 ],
            [ 0.58359754,  4.6000338 ],
            [-0.59114265,  4.019289  ],
            [ 0.4113918 ,  2.4831705 ],
            [ 1.3877766 ,  2.5606558 ],
            [-0.4066615 ,  2.3010864 ],
            [ 0.53182524,  4.5484734 ],
            [ 0.5476439 ,  3.8957043 ],
            [-0.44016668,  4.2665353 ],
            [ 0.06540618,  1.9770341 ],
            [-0.44527835,  2.79227   ],
            [ 0.1853226 ,  2.341187  ],
            [ 0.7631004 ,  3.052445  ],
            [ 0.06769427,  4.2095046 ],
            [-0.234777  ,  4.3941665 ],
            [-0.80452645,  2.8808823 ],
            [-0.21663386,  3.8711824 ],
            [-0.45481792,  3.127202  ],
            [ 0.6813676 ,  3.9301512 ],
            [-0.02085217,  2.858958  ],
            [-0.77275085,  1.4155594 ],
            [ 0.21170408,  2.5074565 ],
            [-0.6096052 ,  3.2142673 ],
            [ 0.6286835 ,  2.595309  ],
            [-1.5694705 ,  1.6471096 ],
            [-0.9350603 ,  3.8853846 ],
            [ 0.5504711 ,  2.39596   ],
            [ 0.614715  ,  2.5657723 ],
            [-0.04280939,  2.3600953 ],
            [ 0.81765246,  2.4454157 ],
            [-0.35487476,  3.840568  ],
            [ 1.3129084 ,  2.799836  ],
            [ 1.2957257 ,  3.7324014 ],
            [ 0.21081987,  3.7207868 ],
            [-1.2219485 ,  1.5647888 ],
            [-1.5206212 ,  2.1671724 ],
            [-0.12463187,  2.856854  ],
            [ 0.19222555,  3.6286852 ],
            [ 0.7549215 ,  3.3364131 ],
            [-0.21568361,  3.7116737 ],
            [-0.43359905,  3.3622594 ],
            [-0.80462474,  2.3572843 ],
            [ 2.1009529 ,  3.1306639 ],
            [ 1.1595324 ,  2.770774  ],
            [-0.4931188 ,  3.3116958 ],
            [-0.33913192,  2.674049  ],
            [-0.42083812,  2.866513  ],
            [ 0.10560735,  1.2475123 ],
            [ 0.89682347,  3.8794355 ],
            [-1.3472564 ,  0.7678237 ],
            [ 1.29582   ,  4.1995854 ],
            [-0.44267592,  3.2680128 ],
            [-1.4850967 ,  3.6658542 ],
            [ 1.6445867 ,  4.1669974 ],
            [ 0.39686966,  2.8328876 ],
            [ 1.2049758 ,  4.1154943 ],
            [ 0.2329063 ,  3.2219486 ],
            [ 0.98941505,  2.6394854 ],
            [ 0.8386533 ,  4.923409  ],
            [ 1.4159845 ,  3.4892316 ],
            [ 0.03147054,  2.958451  ],
            [ 1.2485881 ,  4.2131734 ],
            [ 1.0417484 ,  3.420746  ],
            [ 0.34944412,  1.9527359 ],
            [ 1.0666001 ,  1.0448306 ],
            [-1.8675996 ,  2.6684077 ],
            [ 0.961952  ,  3.0640275 ],
            [ 1.1833825 ,  3.1582932 ],
            [-1.02328   ,  1.4057444 ],
            [ 0.6035042 ,  4.0283585 ],
            [ 1.4228338 ,  3.8163633 ],
            [-0.19441035,  2.0487962 ],
            [ 0.07422742,  3.1779172 ],
            [ 1.1445094 ,  4.2026944 ],
            [ 1.4966571 ,  3.9725122 ],
            [ 0.10014264,  2.586527  ],
            [ 0.01716895,  2.9761229 ],
            [-0.40134135,  2.6361563 ],
            [-0.8921947 ,  2.9560194 ],
            [ 1.6371847 ,  3.4151359 ],
            [-1.430026  ,  3.1877294 ],
            [ 1.1259433 ,  5.9970665 ],
            [-2.0950599 ,  1.6324173 ],
            [ 1.6014951 ,  4.6775293 ],
            [ 0.19590549,  3.015738  ],
            [-0.19311003,  1.2175571 ],
            [-0.22257052,  3.9766564 ],
            [ 0.7019713 ,  3.5201108 ],
            [-0.3225968 ,  3.4080396 ],
            [ 0.1850599 ,  2.0989368 ],
            [ 2.040577  ,  2.544058  ],
            [ 0.35953036,  2.983996  ],
            [ 1.5000627 ,  3.394865  ],
            [-0.02001924,  1.8177806 ],
            [ 0.6084913 ,  4.087865  ],
            [-0.05247078,  2.8531036 ],
            [ 2.1325917 ,  4.3253827 ],
            [ 0.96282464,  3.0731583 ],
            [ 0.33242285,  4.0261064 ],
            [ 0.0260964 ,  1.9833332 ]], dtype=float32),
     array([[ 5.47278583e-01,  2.72134733e+00],
            [-1.60670578e+00,  3.24014497e+00],
            [ 7.21152186e-01,  2.61640286e+00],
            [-1.46102309e+00,  8.20773184e-01],
            [-7.74187565e-01,  2.04132485e+00],
            [-3.70081365e-01,  3.09597874e+00],
            [-1.46450803e-01,  1.69170630e+00],
            [ 7.71030128e-01,  4.01347303e+00],
            [-1.09634727e-01,  3.64303255e+00],
            [-4.05911356e-01,  3.44445181e+00],
            [-4.25650924e-01,  2.97071171e+00],
            [-2.47397438e-01,  3.23278832e+00],
            [-1.49143195e+00,  2.22900534e+00],
            [-6.28680165e-04,  2.06793594e+00],
            [ 1.59795022e+00,  3.56039929e+00],
            [ 8.75499427e-01,  5.36261845e+00],
            [ 3.91845942e-01,  4.04525661e+00],
            [ 1.09663177e+00,  4.39985418e+00],
            [ 1.27118611e+00,  1.71230078e+00],
            [ 1.07768726e+00,  3.99291492e+00],
            [ 7.75416613e-01,  4.06828785e+00],
            [ 4.78489518e-01,  2.18659735e+00],
            [ 1.85532236e+00,  4.50483179e+00],
            [ 3.85746330e-01,  2.01449466e+00],
            [ 1.33948147e+00,  3.18946719e+00],
            [ 4.91453320e-01,  3.20656323e+00],
            [-1.15840411e+00,  1.58438718e+00],
            [ 9.59508896e-01,  3.69362688e+00],
            [-4.96175081e-01,  2.31182361e+00],
            [-1.32516694e+00,  1.53361285e+00],
            [-1.12667158e-01,  2.21807313e+00],
            [-8.17695618e-01,  2.61886096e+00],
            [-4.76714104e-01,  1.42464662e+00],
            [ 1.02416182e+00,  4.32508707e+00],
            [ 3.05084080e-01,  2.85255837e+00],
            [-1.41785836e+00,  2.78842521e+00],
            [-1.26238453e+00,  2.43015504e+00],
            [ 9.78169858e-01,  4.31616354e+00],
            [-9.95135963e-01,  2.58529234e+00],
            [ 1.77876806e+00,  4.50146675e+00],
            [ 9.86418605e-01,  3.76414466e+00],
            [ 8.50796044e-01,  4.99612665e+00],
            [ 8.43003809e-01,  2.49373794e+00],
            [ 1.00520098e+00,  4.61487436e+00],
            [-7.57234097e-01,  1.83389115e+00],
            [-8.74363184e-01,  2.11154270e+00],
            [-4.83681969e-02,  4.26920509e+00],
            [-1.04104206e-01,  3.49317575e+00],
            [ 5.30742884e-01,  2.36709571e+00],
            [-1.09570539e+00,  2.50842166e+00],
            [-4.43705171e-01,  1.64019394e+00],
            [ 3.49012375e-01,  3.02315307e+00],
            [-2.78289604e+00,  4.75962996e-01],
            [-1.21605717e-01,  2.50601864e+00],
            [ 2.02707100e+00,  4.16454935e+00],
            [-1.12730587e+00,  2.17778826e+00],
            [ 9.35182512e-01,  3.54037118e+00],
            [ 1.65620792e+00,  2.67648196e+00],
            [-2.51378131e+00,  3.27776742e+00],
            [ 4.12245780e-01,  3.10756183e+00],
            [ 1.12896606e-01,  3.07394600e+00],
            [-5.76717257e-01,  6.41360521e-01],
            [-3.75410944e-01,  2.48743510e+00],
            [-4.40868825e-01,  2.79482269e+00],
            [ 1.28813207e-01,  2.69578433e+00],
            [ 6.91303849e-01,  2.60120654e+00],
            [-8.68377090e-01,  1.51304626e+00],
            [-3.28325480e-01,  2.90508413e+00],
            [-1.65402651e+00,  2.45638585e+00],
            [-3.44768345e-01,  3.14479351e+00],
            [-1.64355546e-01,  2.52893496e+00],
            [-2.11734489e-01,  2.36082506e+00],
            [ 4.21602696e-01,  3.26668477e+00],
            [ 7.46418893e-01,  3.71823788e+00],
            [ 4.66994792e-01,  3.02948475e+00],
            [-1.14561796e+00,  3.02392602e+00],
            [-1.01831508e+00,  2.70106936e+00],
            [ 8.40311646e-01,  3.93927670e+00],
            [ 1.59380794e+00,  3.06152058e+00],
            [-1.21774346e-01,  4.01543856e+00],
            [ 7.96276629e-01,  3.59320235e+00],
            [-2.56954521e-01,  3.65322113e+00],
            [-4.00427170e-03,  2.94932079e+00],
            [ 2.33644724e+00,  5.23351860e+00],
            [ 2.29875773e-01,  4.48880196e+00],
            [-1.34256732e+00,  3.03134847e+00],
            [-3.62941563e-01,  4.20227194e+00],
            [-5.27580202e-01,  4.19374084e+00],
            [ 3.49907607e-01,  3.08884192e+00],
            [ 3.38893175e-01,  3.43727374e+00],
            [-5.95994473e-01,  4.49286509e+00],
            [ 1.64696717e+00,  3.37063336e+00],
            [-4.74671960e-01,  3.72200823e+00],
            [ 1.58153713e+00,  3.98410940e+00],
            [ 2.23190308e+00,  2.10402036e+00],
            [-1.39981285e-01,  3.10728073e+00],
            [ 5.69586456e-01,  3.84047961e+00],
            [ 1.78660381e+00,  3.77313137e+00],
            [-1.21734464e+00,  2.80100417e+00],
            [-1.97611853e-01,  2.12106133e+00]], dtype=float32)]



`np.vsplit()`함수를 이용하여 입력 데이터셋을 20개로 쪼갭니다. 즉, 미니 배치 당 데이터 샘플이 100개입니다.(`2000 나누기 20 은 100`이므로) 

레이블도 마찬가지로 쪼개줍니다.


```python
mini_label = np.vsplit(targets, 20)

mini_label[:3]
```




    [array([[0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.]], dtype=float32), array([[0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.]], dtype=float32), array([[0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.]], dtype=float32)]



앞의 3개 미니배치에 대해 레이블은 확인해 보았습니다. 앞의 샘플이라 모든 레이블이 0(음성)인 것을 알 수 있습니다.

### **미니 배치 경사하강법 구현**

2중 for문을 사용하여 구현합니다.
* 첫 번째 for문 : 에포크 즉, 전체 데이터셋을 몇번 반복하는지 지정
* 두 번째 for문 : 각각의 미니배치에 대한 반복


* 일정 훈련 단계 번째마다 손실값과 정확도가 출력되게 하였습니다.


```python
for step in range(100):
  for mini_in, mini_tar in zip(mini_batch, mini_label) :
    loss = training_step(mini_in, mini_tar)
    accuracy = my_accuracy(mini_in, mini_tar)
    if step % 10 == 0:
        print(f"Loss at step {step}: {loss:.4f}, Accuracy at step {step}: {accuracy:.4f}")
```

    Loss at step 0: 0.6209, Accuracy at step 0: 1.0000
    Loss at step 0: 0.2809, Accuracy at step 0: 1.0000
    Loss at step 0: 0.0010, Accuracy at step 0: 1.0000
    Loss at step 0: 0.0008, Accuracy at step 0: 1.0000
    Loss at step 0: 0.0007, Accuracy at step 0: 1.0000
    Loss at step 0: 0.0007, Accuracy at step 0: 1.0000
    Loss at step 0: 0.0007, Accuracy at step 0: 1.0000
    Loss at step 0: 0.0005, Accuracy at step 0: 1.0000
    Loss at step 0: 0.0005, Accuracy at step 0: 1.0000
    Loss at step 0: 0.0004, Accuracy at step 0: 1.0000
    Loss at step 0: 0.9322, Accuracy at step 0: 0.9400
    Loss at step 0: 0.2077, Accuracy at step 0: 0.7700
    Loss at step 0: 0.1659, Accuracy at step 0: 0.8900
    Loss at step 0: 0.1335, Accuracy at step 0: 0.8700
    Loss at step 0: 0.1121, Accuracy at step 0: 0.8700
    Loss at step 0: 0.1090, Accuracy at step 0: 0.8900
    Loss at step 0: 0.0844, Accuracy at step 0: 0.9400
    Loss at step 0: 0.0901, Accuracy at step 0: 0.9200
    Loss at step 0: 0.0719, Accuracy at step 0: 0.9600
    Loss at step 0: 0.0588, Accuracy at step 0: 0.9700
    Loss at step 10: 0.0288, Accuracy at step 10: 1.0000
    Loss at step 10: 0.0165, Accuracy at step 10: 1.0000
    Loss at step 10: 0.0100, Accuracy at step 10: 1.0000
    Loss at step 10: 0.0044, Accuracy at step 10: 1.0000
    Loss at step 10: 0.0047, Accuracy at step 10: 1.0000
    Loss at step 10: 0.0013, Accuracy at step 10: 1.0000
    Loss at step 10: 0.0022, Accuracy at step 10: 1.0000
    Loss at step 10: 0.0014, Accuracy at step 10: 1.0000
    Loss at step 10: 0.0013, Accuracy at step 10: 1.0000
    Loss at step 10: 0.0005, Accuracy at step 10: 1.0000
    Loss at step 10: 0.0822, Accuracy at step 10: 0.9900
    Loss at step 10: 0.0601, Accuracy at step 10: 0.9700
    Loss at step 10: 0.0643, Accuracy at step 10: 0.9700
    Loss at step 10: 0.0482, Accuracy at step 10: 0.9900
    Loss at step 10: 0.0477, Accuracy at step 10: 0.9800
    Loss at step 10: 0.0488, Accuracy at step 10: 0.9800
    Loss at step 10: 0.0552, Accuracy at step 10: 0.9700
    Loss at step 10: 0.0499, Accuracy at step 10: 0.9900
    Loss at step 10: 0.0529, Accuracy at step 10: 0.9700
    Loss at step 10: 0.0424, Accuracy at step 10: 0.9800
    Loss at step 20: 0.0232, Accuracy at step 20: 1.0000
    Loss at step 20: 0.0121, Accuracy at step 20: 1.0000
    Loss at step 20: 0.0071, Accuracy at step 20: 1.0000
    Loss at step 20: 0.0023, Accuracy at step 20: 1.0000
    Loss at step 20: 0.0042, Accuracy at step 20: 1.0000
    Loss at step 20: 0.0007, Accuracy at step 20: 1.0000
    Loss at step 20: 0.0025, Accuracy at step 20: 1.0000
    Loss at step 20: 0.0023, Accuracy at step 20: 1.0000
    Loss at step 20: 0.0027, Accuracy at step 20: 1.0000
    Loss at step 20: 0.0008, Accuracy at step 20: 1.0000
    Loss at step 20: 0.0690, Accuracy at step 20: 0.9900
    Loss at step 20: 0.0441, Accuracy at step 20: 0.9900
    Loss at step 20: 0.0479, Accuracy at step 20: 1.0000
    Loss at step 20: 0.0356, Accuracy at step 20: 1.0000
    Loss at step 20: 0.0329, Accuracy at step 20: 1.0000
    Loss at step 20: 0.0337, Accuracy at step 20: 1.0000
    Loss at step 20: 0.0428, Accuracy at step 20: 0.9900
    Loss at step 20: 0.0337, Accuracy at step 20: 1.0000
    Loss at step 20: 0.0391, Accuracy at step 20: 0.9900
    Loss at step 20: 0.0333, Accuracy at step 20: 1.0000
    Loss at step 30: 0.0265, Accuracy at step 30: 1.0000
    Loss at step 30: 0.0121, Accuracy at step 30: 1.0000
    Loss at step 30: 0.0065, Accuracy at step 30: 1.0000
    Loss at step 30: 0.0018, Accuracy at step 30: 1.0000
    Loss at step 30: 0.0050, Accuracy at step 30: 1.0000
    Loss at step 30: 0.0010, Accuracy at step 30: 1.0000
    Loss at step 30: 0.0033, Accuracy at step 30: 1.0000
    Loss at step 30: 0.0034, Accuracy at step 30: 1.0000
    Loss at step 30: 0.0043, Accuracy at step 30: 1.0000
    Loss at step 30: 0.0016, Accuracy at step 30: 1.0000
    Loss at step 30: 0.0719, Accuracy at step 30: 1.0000
    Loss at step 30: 0.0404, Accuracy at step 30: 1.0000
    Loss at step 30: 0.0416, Accuracy at step 30: 1.0000
    Loss at step 30: 0.0309, Accuracy at step 30: 1.0000
    Loss at step 30: 0.0271, Accuracy at step 30: 1.0000
    Loss at step 30: 0.0274, Accuracy at step 30: 1.0000
    Loss at step 30: 0.0370, Accuracy at step 30: 0.9900
    Loss at step 30: 0.0258, Accuracy at step 30: 1.0000
    Loss at step 30: 0.0315, Accuracy at step 30: 0.9900
    Loss at step 30: 0.0282, Accuracy at step 30: 1.0000
    Loss at step 40: 0.0280, Accuracy at step 40: 1.0000
    Loss at step 40: 0.0124, Accuracy at step 40: 1.0000
    Loss at step 40: 0.0064, Accuracy at step 40: 1.0000
    Loss at step 40: 0.0018, Accuracy at step 40: 1.0000
    Loss at step 40: 0.0055, Accuracy at step 40: 1.0000
    Loss at step 40: 0.0011, Accuracy at step 40: 1.0000
    Loss at step 40: 0.0035, Accuracy at step 40: 1.0000
    Loss at step 40: 0.0039, Accuracy at step 40: 1.0000
    Loss at step 40: 0.0048, Accuracy at step 40: 1.0000
    Loss at step 40: 0.0018, Accuracy at step 40: 1.0000
    Loss at step 40: 0.0746, Accuracy at step 40: 1.0000
    Loss at step 40: 0.0387, Accuracy at step 40: 1.0000
    Loss at step 40: 0.0393, Accuracy at step 40: 1.0000
    Loss at step 40: 0.0293, Accuracy at step 40: 1.0000
    Loss at step 40: 0.0250, Accuracy at step 40: 1.0000
    Loss at step 40: 0.0255, Accuracy at step 40: 1.0000
    Loss at step 40: 0.0351, Accuracy at step 40: 0.9900
    Loss at step 40: 0.0236, Accuracy at step 40: 1.0000
    Loss at step 40: 0.0294, Accuracy at step 40: 0.9900
    Loss at step 40: 0.0267, Accuracy at step 40: 1.0000
    Loss at step 50: 0.0284, Accuracy at step 50: 1.0000
    Loss at step 50: 0.0131, Accuracy at step 50: 1.0000
    Loss at step 50: 0.0066, Accuracy at step 50: 1.0000
    Loss at step 50: 0.0019, Accuracy at step 50: 1.0000
    Loss at step 50: 0.0057, Accuracy at step 50: 1.0000
    Loss at step 50: 0.0012, Accuracy at step 50: 1.0000
    Loss at step 50: 0.0037, Accuracy at step 50: 1.0000
    Loss at step 50: 0.0042, Accuracy at step 50: 1.0000
    Loss at step 50: 0.0052, Accuracy at step 50: 1.0000
    Loss at step 50: 0.0020, Accuracy at step 50: 1.0000
    Loss at step 50: 0.0752, Accuracy at step 50: 1.0000
    Loss at step 50: 0.0373, Accuracy at step 50: 1.0000
    Loss at step 50: 0.0380, Accuracy at step 50: 1.0000
    Loss at step 50: 0.0285, Accuracy at step 50: 1.0000
    Loss at step 50: 0.0241, Accuracy at step 50: 1.0000
    Loss at step 50: 0.0246, Accuracy at step 50: 1.0000
    Loss at step 50: 0.0344, Accuracy at step 50: 0.9900
    Loss at step 50: 0.0229, Accuracy at step 50: 1.0000
    Loss at step 50: 0.0287, Accuracy at step 50: 0.9900
    Loss at step 50: 0.0263, Accuracy at step 50: 1.0000
    Loss at step 60: 0.0281, Accuracy at step 60: 1.0000
    Loss at step 60: 0.0134, Accuracy at step 60: 1.0000
    Loss at step 60: 0.0067, Accuracy at step 60: 1.0000
    Loss at step 60: 0.0020, Accuracy at step 60: 1.0000
    Loss at step 60: 0.0059, Accuracy at step 60: 1.0000
    Loss at step 60: 0.0013, Accuracy at step 60: 1.0000
    Loss at step 60: 0.0038, Accuracy at step 60: 1.0000
    Loss at step 60: 0.0043, Accuracy at step 60: 1.0000
    Loss at step 60: 0.0054, Accuracy at step 60: 1.0000
    Loss at step 60: 0.0021, Accuracy at step 60: 1.0000
    Loss at step 60: 0.0754, Accuracy at step 60: 1.0000
    Loss at step 60: 0.0364, Accuracy at step 60: 1.0000
    Loss at step 60: 0.0373, Accuracy at step 60: 1.0000
    Loss at step 60: 0.0282, Accuracy at step 60: 1.0000
    Loss at step 60: 0.0237, Accuracy at step 60: 1.0000
    Loss at step 60: 0.0242, Accuracy at step 60: 1.0000
    Loss at step 60: 0.0341, Accuracy at step 60: 0.9900
    Loss at step 60: 0.0227, Accuracy at step 60: 1.0000
    Loss at step 60: 0.0286, Accuracy at step 60: 0.9900
    Loss at step 60: 0.0263, Accuracy at step 60: 1.0000
    Loss at step 70: 0.0275, Accuracy at step 70: 1.0000
    Loss at step 70: 0.0138, Accuracy at step 70: 1.0000
    Loss at step 70: 0.0068, Accuracy at step 70: 1.0000
    Loss at step 70: 0.0021, Accuracy at step 70: 1.0000
    Loss at step 70: 0.0061, Accuracy at step 70: 1.0000
    Loss at step 70: 0.0013, Accuracy at step 70: 1.0000
    Loss at step 70: 0.0038, Accuracy at step 70: 1.0000
    Loss at step 70: 0.0045, Accuracy at step 70: 1.0000
    Loss at step 70: 0.0055, Accuracy at step 70: 1.0000
    Loss at step 70: 0.0021, Accuracy at step 70: 1.0000
    Loss at step 70: 0.0747, Accuracy at step 70: 1.0000
    Loss at step 70: 0.0358, Accuracy at step 70: 1.0000
    Loss at step 70: 0.0369, Accuracy at step 70: 1.0000
    Loss at step 70: 0.0280, Accuracy at step 70: 1.0000
    Loss at step 70: 0.0234, Accuracy at step 70: 1.0000
    Loss at step 70: 0.0241, Accuracy at step 70: 1.0000
    Loss at step 70: 0.0341, Accuracy at step 70: 0.9900
    Loss at step 70: 0.0227, Accuracy at step 70: 1.0000
    Loss at step 70: 0.0286, Accuracy at step 70: 0.9900
    Loss at step 70: 0.0264, Accuracy at step 70: 1.0000
    Loss at step 80: 0.0271, Accuracy at step 80: 1.0000
    Loss at step 80: 0.0140, Accuracy at step 80: 1.0000
    Loss at step 80: 0.0069, Accuracy at step 80: 1.0000
    Loss at step 80: 0.0022, Accuracy at step 80: 1.0000
    Loss at step 80: 0.0062, Accuracy at step 80: 1.0000
    Loss at step 80: 0.0014, Accuracy at step 80: 1.0000
    Loss at step 80: 0.0039, Accuracy at step 80: 1.0000
    Loss at step 80: 0.0045, Accuracy at step 80: 1.0000
    Loss at step 80: 0.0056, Accuracy at step 80: 1.0000
    Loss at step 80: 0.0022, Accuracy at step 80: 1.0000
    Loss at step 80: 0.0745, Accuracy at step 80: 1.0000
    Loss at step 80: 0.0353, Accuracy at step 80: 1.0000
    Loss at step 80: 0.0366, Accuracy at step 80: 1.0000
    Loss at step 80: 0.0279, Accuracy at step 80: 1.0000
    Loss at step 80: 0.0233, Accuracy at step 80: 1.0000
    Loss at step 80: 0.0240, Accuracy at step 80: 1.0000
    Loss at step 80: 0.0340, Accuracy at step 80: 0.9900
    Loss at step 80: 0.0227, Accuracy at step 80: 1.0000
    Loss at step 80: 0.0286, Accuracy at step 80: 0.9900
    Loss at step 80: 0.0265, Accuracy at step 80: 1.0000
    Loss at step 90: 0.0268, Accuracy at step 90: 1.0000
    Loss at step 90: 0.0139, Accuracy at step 90: 1.0000
    Loss at step 90: 0.0070, Accuracy at step 90: 1.0000
    Loss at step 90: 0.0022, Accuracy at step 90: 1.0000
    Loss at step 90: 0.0062, Accuracy at step 90: 1.0000
    Loss at step 90: 0.0014, Accuracy at step 90: 1.0000
    Loss at step 90: 0.0039, Accuracy at step 90: 1.0000
    Loss at step 90: 0.0046, Accuracy at step 90: 1.0000
    Loss at step 90: 0.0057, Accuracy at step 90: 1.0000
    Loss at step 90: 0.0023, Accuracy at step 90: 1.0000
    Loss at step 90: 0.0742, Accuracy at step 90: 1.0000
    Loss at step 90: 0.0349, Accuracy at step 90: 1.0000
    Loss at step 90: 0.0364, Accuracy at step 90: 1.0000
    Loss at step 90: 0.0278, Accuracy at step 90: 1.0000
    Loss at step 90: 0.0232, Accuracy at step 90: 1.0000
    Loss at step 90: 0.0239, Accuracy at step 90: 1.0000
    Loss at step 90: 0.0340, Accuracy at step 90: 0.9900
    Loss at step 90: 0.0227, Accuracy at step 90: 1.0000
    Loss at step 90: 0.0287, Accuracy at step 90: 0.9900
    Loss at step 90: 0.0266, Accuracy at step 90: 1.0000
    

10번째 에포크마다 미니배치들의 손실값과 정확도를 출력하도록 하였습니다.

여기서 주의 한 것은 두번째 for문을 쓸 때 `zip`을 통해 미니배치의 데이터 샘플과 레이블을 묶어서 해주었다는 것입니다.

10번의 에포크마다 미니배치의 손실값과 정확도를 출력하도록 하였기에 출력되는 값이 `100 곱하기 10`이므로 천개의 손실값이 출력됩니다.


```python
my_accuracy(inputs, targets)
```




    0.996



정확도는 다음과 같습니다.

에포크를 500회를 더 늘려서 훈련을 더해 보겠습니다.


```python
for step in range(500):
  for mini_in, mini_tar in zip(mini_batch, mini_label) :
    loss = training_step(mini_in, mini_tar)
    accuracy = my_accuracy(mini_in, mini_tar)
    if step % 100 == 0:
        print(f"Loss at step {step}: {loss:.4f}, Accuracy at step {step}: {accuracy:.4f}")
```

    Loss at step 0: 0.0265, Accuracy at step 0: 1.0000
    Loss at step 0: 0.0139, Accuracy at step 0: 1.0000
    Loss at step 0: 0.0070, Accuracy at step 0: 1.0000
    Loss at step 0: 0.0022, Accuracy at step 0: 1.0000
    Loss at step 0: 0.0063, Accuracy at step 0: 1.0000
    Loss at step 0: 0.0014, Accuracy at step 0: 1.0000
    Loss at step 0: 0.0039, Accuracy at step 0: 1.0000
    Loss at step 0: 0.0046, Accuracy at step 0: 1.0000
    Loss at step 0: 0.0057, Accuracy at step 0: 1.0000
    Loss at step 0: 0.0023, Accuracy at step 0: 1.0000
    Loss at step 0: 0.0739, Accuracy at step 0: 1.0000
    Loss at step 0: 0.0346, Accuracy at step 0: 1.0000
    Loss at step 0: 0.0363, Accuracy at step 0: 1.0000
    Loss at step 0: 0.0278, Accuracy at step 0: 1.0000
    Loss at step 0: 0.0232, Accuracy at step 0: 1.0000
    Loss at step 0: 0.0239, Accuracy at step 0: 1.0000
    Loss at step 0: 0.0340, Accuracy at step 0: 0.9900
    Loss at step 0: 0.0227, Accuracy at step 0: 1.0000
    Loss at step 0: 0.0288, Accuracy at step 0: 0.9900
    Loss at step 0: 0.0267, Accuracy at step 0: 1.0000
    Loss at step 100: 0.0247, Accuracy at step 100: 1.0000
    Loss at step 100: 0.0143, Accuracy at step 100: 1.0000
    Loss at step 100: 0.0074, Accuracy at step 100: 1.0000
    Loss at step 100: 0.0025, Accuracy at step 100: 1.0000
    Loss at step 100: 0.0066, Accuracy at step 100: 1.0000
    Loss at step 100: 0.0016, Accuracy at step 100: 1.0000
    Loss at step 100: 0.0041, Accuracy at step 100: 1.0000
    Loss at step 100: 0.0050, Accuracy at step 100: 1.0000
    Loss at step 100: 0.0061, Accuracy at step 100: 1.0000
    Loss at step 100: 0.0025, Accuracy at step 100: 1.0000
    Loss at step 100: 0.0721, Accuracy at step 100: 1.0000
    Loss at step 100: 0.0329, Accuracy at step 100: 1.0000
    Loss at step 100: 0.0356, Accuracy at step 100: 1.0000
    Loss at step 100: 0.0275, Accuracy at step 100: 1.0000
    Loss at step 100: 0.0228, Accuracy at step 100: 1.0000
    Loss at step 100: 0.0237, Accuracy at step 100: 1.0000
    Loss at step 100: 0.0341, Accuracy at step 100: 0.9900
    Loss at step 100: 0.0229, Accuracy at step 100: 1.0000
    Loss at step 100: 0.0291, Accuracy at step 100: 0.9900
    Loss at step 100: 0.0272, Accuracy at step 100: 1.0000
    Loss at step 200: 0.0239, Accuracy at step 200: 1.0000
    Loss at step 200: 0.0143, Accuracy at step 200: 1.0000
    Loss at step 200: 0.0075, Accuracy at step 200: 1.0000
    Loss at step 200: 0.0026, Accuracy at step 200: 1.0000
    Loss at step 200: 0.0068, Accuracy at step 200: 1.0000
    Loss at step 200: 0.0016, Accuracy at step 200: 1.0000
    Loss at step 200: 0.0042, Accuracy at step 200: 1.0000
    Loss at step 200: 0.0051, Accuracy at step 200: 1.0000
    Loss at step 200: 0.0062, Accuracy at step 200: 1.0000
    Loss at step 200: 0.0027, Accuracy at step 200: 1.0000
    Loss at step 200: 0.0714, Accuracy at step 200: 1.0000
    Loss at step 200: 0.0323, Accuracy at step 200: 1.0000
    Loss at step 200: 0.0354, Accuracy at step 200: 1.0000
    Loss at step 200: 0.0274, Accuracy at step 200: 1.0000
    Loss at step 200: 0.0227, Accuracy at step 200: 1.0000
    Loss at step 200: 0.0237, Accuracy at step 200: 1.0000
    Loss at step 200: 0.0342, Accuracy at step 200: 0.9900
    Loss at step 200: 0.0231, Accuracy at step 200: 1.0000
    Loss at step 200: 0.0293, Accuracy at step 200: 0.9900
    Loss at step 200: 0.0274, Accuracy at step 200: 1.0000
    Loss at step 300: 0.0234, Accuracy at step 300: 1.0000
    Loss at step 300: 0.0142, Accuracy at step 300: 1.0000
    Loss at step 300: 0.0075, Accuracy at step 300: 1.0000
    Loss at step 300: 0.0026, Accuracy at step 300: 1.0000
    Loss at step 300: 0.0069, Accuracy at step 300: 1.0000
    Loss at step 300: 0.0017, Accuracy at step 300: 1.0000
    Loss at step 300: 0.0043, Accuracy at step 300: 1.0000
    Loss at step 300: 0.0052, Accuracy at step 300: 1.0000
    Loss at step 300: 0.0063, Accuracy at step 300: 1.0000
    Loss at step 300: 0.0027, Accuracy at step 300: 1.0000
    Loss at step 300: 0.0712, Accuracy at step 300: 1.0000
    Loss at step 300: 0.0320, Accuracy at step 300: 1.0000
    Loss at step 300: 0.0353, Accuracy at step 300: 1.0000
    Loss at step 300: 0.0274, Accuracy at step 300: 1.0000
    Loss at step 300: 0.0227, Accuracy at step 300: 1.0000
    Loss at step 300: 0.0237, Accuracy at step 300: 1.0000
    Loss at step 300: 0.0342, Accuracy at step 300: 0.9900
    Loss at step 300: 0.0231, Accuracy at step 300: 1.0000
    Loss at step 300: 0.0294, Accuracy at step 300: 0.9900
    Loss at step 300: 0.0276, Accuracy at step 300: 1.0000
    Loss at step 400: 0.0231, Accuracy at step 400: 1.0000
    Loss at step 400: 0.0143, Accuracy at step 400: 1.0000
    Loss at step 400: 0.0076, Accuracy at step 400: 1.0000
    Loss at step 400: 0.0027, Accuracy at step 400: 1.0000
    Loss at step 400: 0.0070, Accuracy at step 400: 1.0000
    Loss at step 400: 0.0017, Accuracy at step 400: 1.0000
    Loss at step 400: 0.0043, Accuracy at step 400: 1.0000
    Loss at step 400: 0.0052, Accuracy at step 400: 1.0000
    Loss at step 400: 0.0064, Accuracy at step 400: 1.0000
    Loss at step 400: 0.0028, Accuracy at step 400: 1.0000
    Loss at step 400: 0.0709, Accuracy at step 400: 1.0000
    Loss at step 400: 0.0319, Accuracy at step 400: 1.0000
    Loss at step 400: 0.0352, Accuracy at step 400: 1.0000
    Loss at step 400: 0.0274, Accuracy at step 400: 1.0000
    Loss at step 400: 0.0227, Accuracy at step 400: 1.0000
    Loss at step 400: 0.0237, Accuracy at step 400: 1.0000
    Loss at step 400: 0.0343, Accuracy at step 400: 0.9900
    Loss at step 400: 0.0232, Accuracy at step 400: 1.0000
    Loss at step 400: 0.0295, Accuracy at step 400: 0.9900
    Loss at step 400: 0.0277, Accuracy at step 400: 1.0000
    

## **예측**
---
실제로 훈련된 모델을 이용하여 예측한 것을 이미지로 확인해보고 앞의 그림과 비교해볼 수 있습니다. 


```python
predictions = model(inputs)
```

일단 훈련된 모델에 입력데이터를 넣어서 예측합니다.

그리고 0.5보다 크면 양성으로, 작으면 음성으로 판정하게 하였습니다.


```python
plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
plt.show()
```


![png](output_70_0.png)


앞의 그림과 비교해보았을 때 전체적으로 비슷한 양상을 보이고 있습니다. 세밀하게 따졌을 때는 경계부분이 다르긴 합니다.


```python
my_accuracy(inputs, targets)
```




    0.9965



정확도를 확인해보니 99%의 성능을 발휘하고 있습니다.

## **궁금증 생긴 부분**
---
1. Q.코드를 런타임을 초기화하고 여러번 계속 해보았는데, 코드는 수정하지 않았는데 할 때마다 성능(정확도)가 다르게 나탔습니다. 시간에 따라 훈련에 영향을 미치는 것 같은데, 어떤 부분에서 시간과 관련이 된 건지 모르겠습니다.

1. A.혹시 가중치를 처음에 초기화하는 작업에서 random으로 초기화 하는 함수 부분이 시간과 관련이 있는 것이 아닐까 생각됩니다.

2. Q.모델의 성능을 개선시키기 위해 각 층의 출력 특성수(즉, 유닛)를 계속 임의로 바꿔가면서 모델을 훈련해보았습니다. 즉, inter_layers_dim1과 inter_layers_dim2를 바꾸어 주면서 해보았는데 10개, 5개, 4개, 20개 등을 해보았는데 그러면 모든 샘플을 양성으로 판단하여 정확도가 0.5정도를 보였습니다.

2. A. 이유를 생각해보려 했으나 잘 떠오르지 않았습니다. 입력 데이터의 특성이 2개인데 더 많은 수의 특성으로 바꾸니 데이터가 부풀려져서 값이 결국 커져서 0.5보다 큰 값을 반환하여 결국 양성으로 판단하는 것인지 확신이 서지 않습니다.
