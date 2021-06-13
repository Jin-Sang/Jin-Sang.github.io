---
layout: single
title:  "My ExtraTreesClassifier"

use_math: true

---


# ***문제 1 : 엑스트라 트리 직접 구현***

먼저 엑스트라 트리에 대해 설명하자면 엑스트라 트리는 랜덤 포레스트와 같이 결정트리 모델을 이용한 배깅 학습을 하는 앙상블 학습 모델이다. 

엑스트라 트리의 핵심은 결정트리 모델이 노드를 분리 할 때 사용할 특성을 임의로 선택할 뿐만 아니라 그 임계값도 모두 해보는 것이 아니라 일부를 무작위로 선택하여 학습 하는 모델을 말한다.
* 훈련에 사용할 특성 무작위 선택 후 그 중 최적 선택
* 임계값 무작위 선택 후 그 중 최적 선택

**랜덤 포레스트 VS 엑스트라 트리**

---



두 개는 전반적으로 비슷한 모델입니다. 차이점은 엑스트라 트리는 특성뿐만 아니라 임곗값도 무작위로 선택해서 그 중 최적을 고른다는 것입니다.


---


결국 유일한 차이는 Base 결정 트리가 다르다는 것 


*   RandomForestClassifier 클래스가 사용하는 결정 트리 : DecisionTreeClassifier
*    ExtraTreesClassifier가 사용하는 결정 트리 : ExtraTreeClassifier ( 두 개 이름 헷갈리지 말 것)




구현 순서
* DecisionTreeClassifier로 ExtraTreeClassifier를 구현
* 구현한 ExtraTreeClassifier로 (DecisionTreeClassifier로  RandomForestClassifier를 구현하는 것과 같이) ExtraTreesClassifier 구현

## 1단계 : DecisionTreeClassifier로 ExtraTreeClassifier를 구현


일단 사이킷런에서 지원하는 moons dataset을 가져오겠습니다 ( make_moons(n_samples=10000, noise=0.4)로 가져옵니다.)


```python
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)
```

(출력의 일정함을 위해 random_state=42 옵션도 추가하였습니다.)


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

훈련 세트와 테스트 세트를 분류하였습니다. 훈련세트 80% 테스트세트 20% 입니다.

DecisionTreeClassifier 클래스는 결정트리 알고리즘을 활용한 분류 모델을 지원한다.

여기서 DecisionTreeClassifer 클래스의 splitter="random"으로 해줌으로써 결정트리를 훈련 할 때 특성을 모두 사용하지 않고, 무작위로 선택하여 그 중 최적을 선택하도록 합니다. 
* 각 노드에서 분할을 할 때 무작위 전략입니다. 무작위로 선택함으로써 특성과 임곗값을 무작위로 선택 후 그 중 최적을 찾는 효과를 냅니다.

max_features="auto" 는 고려할 최대 특성 수를 말하는데 auto는 보통 최대 특성 수의 제곱근만큼 특성이 선택 될 수 있습니다.
* 이것을 통해 고려할 특성 수에 따라 특성들을 숫자만큼 선택합니다.

**splitter="random"과 max_features="auto"을 통해서 특성뿐만 아니라 임곗값도 임의로 선택하는 엑스트라 트리의 핵심을 구현하였습니다.**
* 즉, 특성도 무작위, 임계값도 무작위 선택입니다.


```python
from sklearn.tree import DecisionTreeClassifier    # 사이킷런에서 지원하는 결정트리 분류기 입니다.

from sklearn.model_selection import GridSearchCV


params = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}
grid_search_cv = GridSearchCV(DecisionTreeClassifier(splitter="random", max_features="auto", random_state=42 ), params, verbose=1, cv=3) #ExtraTreeClassifier구현

grid_search_cv.fit(X_train, y_train)
```

    Fitting 3 folds for each of 294 candidates, totalling 882 fits
    

    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done 882 out of 882 | elapsed:    2.6s finished
    




    GridSearchCV(cv=3, error_score=nan,
                 estimator=DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None,
                                                  criterion='gini', max_depth=None,
                                                  max_features='auto',
                                                  max_leaf_nodes=None,
                                                  min_impurity_decrease=0.0,
                                                  min_impurity_split=None,
                                                  min_samples_leaf=1,
                                                  min_samples_split=2,
                                                  min_weight_fraction_leaf=0.0,
                                                  presort='deprecated',
                                                  random_state=42,
                                                  splitter='random'),
                 iid='deprecated', n_jobs=None,
                 param_grid={'max_leaf_nodes': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                                13, 14, 15, 16, 17, 18, 19, 20, 21,
                                                22, 23, 24, 25, 26, 27, 28, 29, 30,
                                                31, ...],
                             'min_samples_split': [2, 3, 4]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring=None, verbose=1)



교차검증을 이용한 그리드 탐색을 이용하여 DecisionTreeClassifier 모델의 최적의 하이퍼파라미터 값을 찾습니다.

* max_leaf_nodes : 최종 분류되는 잎(리프) 노드의 갯수를 의미합니다.
* min_samples_split : 노드를 분할하기 위해 필요한 최소 샘플 수

여기서는 max_leaf_nodes 에 큰 중점을 두고 그리드 탐색을 하여 최적의 파라미터 값을 구하도록 합니다.


```python
grid_search_cv.best_estimator_
```




    DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                           max_depth=None, max_features='auto', max_leaf_nodes=62,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=3,
                           min_weight_fraction_leaf=0.0, presort='deprecated',
                           random_state=42, splitter='random')



그렇게 찾은 최적의 하이퍼파라미터 값을 찾아보니 
* max_leaf_nodes = 62
* min_samples_split = 3
를 확인할 수 있었습니다.

GridSearchCV은 자연스럽게 훈련세트에 대해 찾은 최적의 하이퍼파라미터로 훈련을 시킵니다.
(기본값으로 refit=True 로 설정되기 때문입니다.) 따라서 바로 훈련된 ExtraTreeClassifier(결정트리로 구현한)에 대하여 테스트 세트에 대한 성능 평가를 해보겠습니다.


```python
from sklearn.metrics import accuracy_score

y_pred = grid_search_cv.predict(X_test)
accuracy_score(y_test, y_pred)
```




    0.8485



성능이 약 84.85%로 확인되었습니다.


```python
myExtraTreeClassifier = DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=None, max_features='auto', max_leaf_nodes=62,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=3,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=42, splitter='random')
```

즉 이것이 제가 구현하고 최적의 하이퍼파라미터를 그리드탐색으로 찾은 최적의 ExtraTreeClassifier 분류기입니다. 

사실 DecisionTreeClassifier와 차이점은 splitter='random'과 max_features='auto' 밖에 없습니다.

---
**알아보니 실제로 사이킷런에서 지원하는 ExtraTreeClassifier 클래스는 DecisionTreeClassifier 클래스를 상속받고 위에서 말한 두 가지 차이 외에는 동일합니다.**

### 1.1 사이킷런 지원 모델과 성능비교

사이킷 런에서 지원하는 ExtraTreeClassifier 모델을 이용합니다.


```python
from sklearn.tree import ExtraTreeClassifier

extra_tree = ExtraTreeClassifier(random_state=42)
extra_tree.fit(X_train, y_train)

```




    ExtraTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                        max_depth=None, max_features='auto', max_leaf_nodes=None,
                        min_impurity_decrease=0.0, min_impurity_split=None,
                        min_samples_leaf=1, min_samples_split=2,
                        min_weight_fraction_leaf=0.0, random_state=42,
                        splitter='random')



훈련시켰고 이제 테스트 세트에 대해서 성능을 평가해보겠습니다.


```python
y_pred = extra_tree.predict(X_test)
accuracy_score(y_test, y_pred)
```




    0.8155



약 81.5% 인데, 무작위 선택을 하기 때문에 직접 구현한 것과 차이가 약 3%가량 나는 것으로 보입니다.

## 2단계 : 엑스트라 트리 구현

이번에는 위에서 구한 결정트리를 이용하여 엑스트라 트리 모델을 구현하여 보겠습니다.

먼저 데이타 세트의 부분집합을 1000개 만듭니다. 각각의 부분집합은 무작위로 100개의 훈련 샘플들을 선택하여 가지고 있습니다. (중복 허용) 
* 여기서 사이킷런의 ShuffleSplit을 이용하여 진행하였습니다.


```python
from sklearn.model_selection import ShuffleSplit

n_trees = 1000
n_instances = 100

mini_sets = []

rs = ShuffleSplit(n_splits=n_trees, test_size=len(X_train) - n_instances, random_state=42)
for mini_train_index, mini_test_index in rs.split(X_train):
    X_mini_train = X_train[mini_train_index]
    y_mini_train = y_train[mini_train_index]
    mini_sets.append((X_mini_train, y_mini_train))
```

각각의 만들어진 훈련 부분 집합 샘플세트들을 위에서 찾은 최적의 하이퍼파라미터의 결정트리(직접만든 ExtraTreeClassifier) 모델에 훈련시킵니다. 그리고 그 1000개의 훈련된 결정트리 모델을 테스트 세트로 성능을 평가합니다. 


```python
from sklearn.base import clone

forest = [clone(grid_search_cv.best_estimator_) for _ in range(n_trees)]

accuracy_scores = []

for tree, (X_mini_train, y_mini_train) in zip(forest, mini_sets):
    tree.fit(X_mini_train, y_mini_train)
    
    y_pred = tree.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred))

np.mean(accuracy_scores)
```




    0.7850944999999999



1단계에서는 전체 훈련세트를 이용하였지만, 여기서는 각각의 모델이 더 작은 훈련세트를 이용하였기 때문에 성능이 1단계에서 찾은 모델에 비해 안 좋아 보입니다. ( 약 78%를 보입니다. - 이 수치는 1000개의 결정트리 모델(직접만든 ExtraTreeClassifier)의 성능 수치에 대한 평균입니다.)

자 이제 엑스트라 트리의 최종 예측을 구현하겠습니다.
* 1000개의 훈련된 분류기가 테스트 세트의 각각의 샘플에 대해 예측값을 예측합니다.
* 그리고 최종적으로 각각의 테스트 샘플에 대한 1000개의 예측값 중 최빈값(여기서는 클래스를 2개로 분류하므로 과반수)을 최종적으로 샘플에 대한 랜덤 포레스트 모델의 예측값으로 합니다.


```python
Y_pred = np.empty([n_trees, len(X_test)], dtype=np.uint8)

for tree_index, tree in enumerate(forest):
    Y_pred[tree_index] = tree.predict(X_test)
```

 SciPy의 mode()를 이용해서 과반수 값을 최종 예측값으로 합니다.


```python
from scipy.stats import mode

y_pred_majority_votes, n_votes = mode(Y_pred, axis=0)
```

자 이제 랜덤 포레스트 모델의 테스트 세트에 대한 성능을 평가하여 보겠습니다.


```python
accuracy_score(y_test, y_pred_majority_votes.reshape([-1]))
```




    0.859



약 86%로 약 1.5% 가량 정확도 성능의 향상이 보입니다. 즉, 앙상블 학습인 엑스트라 트리 모델 훈련을 통해 단일의 결정 트리 모델(직접만든 ExtraTreeClassifier)의 성능보다 더 좋아졌다는 것이 확인이 되었습니다. 

## 3단계 : 사이킷런의 엑스트라 트리 모델과 성능비교

---
사이킷 런에서 제공하는 ExtraTreesClassifier 모델을 이용하겠습니다.


```python
from sklearn.ensemble import ExtraTreesClassifier

clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
```




    ExtraTreesClassifier(bootstrap=False, ccp_alpha=0.0, class_weight=None,
                         criterion='gini', max_depth=None, max_features='auto',
                         max_leaf_nodes=None, max_samples=None,
                         min_impurity_decrease=0.0, min_impurity_split=None,
                         min_samples_leaf=1, min_samples_split=2,
                         min_weight_fraction_leaf=0.0, n_estimators=100,
                         n_jobs=None, oob_score=False, random_state=42, verbose=0,
                         warm_start=False)



훈련세트에 대해서 훈련을 시켰습니다. 이제 성능을 확인해 보겠습니다.


```python
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)
```




    0.855



약 85.5%로 약 0.4% 차이밖에 나지 않습니다. 

---
결정트리를 이용하여 최대한 조건을 맞추었기에 비슷한 성능이 나온 것이 아닐까 생각됩니다.
