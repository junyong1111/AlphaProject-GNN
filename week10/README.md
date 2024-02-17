# Attention Is All you Need

<img width="1573" alt="1" src="https://github.com/junyong1111/junyong1111/assets/79856225/36c3e338-2ea6-48b4-91bc-b927987570ba">

Transformer : RNN 기반의 매커니즘이 아닌 Attention 기반의 매커니즘을 이용

### 기존 Seq2Seq 모델의 한계점

- Context Vector의 소스 문장을 모두 압축하므로 병목 현상으로 인한 성능저하를 피하기 힘들다.

<img width="1596" alt="2" src="https://github.com/junyong1111/junyong1111/assets/79856225/db8ee877-97ed-4614-b294-316ce70d5e9c">

단어가 입력될 때마다 이전 단어에 대한 정보를 가진 히든스테이트 값을 갱신해야 하한다. 

이 때 마지막 단어가 들어왔을때  그때에 히든스테이트값은 소스문장을 대표할 수 있는 문맥적인 정보를 가진 Context Vector가 될 수 있다.

—# 위와 같은 고정된 Context Vector를 생성하는게 아닌 소스문장의 전체를 매번 입력값으로 받아보면?

<img width="1371" alt="3" src="https://github.com/junyong1111/junyong1111/assets/79856225/a6272a0f-bc65-4b0d-b206-7376a79103ff">


<img width="1333" alt="4" src="https://github.com/junyong1111/junyong1111/assets/79856225/8cd19224-b374-4756-b83c-f1a854e37979">

- 에너지 : 디코더가 매번 단어를 만들때마다 인코더의 모든 출력을 고려함
    - 디코더가 만든 이전 단어에 대한 값(히든스테이트)과 인코더의 모든 출력((히든스테이트))과 비교하여 가장 **연관이 높은 값을 구함**
- 가중치 : 에너지에서 얻었던 값들을 **소프트맥스를 통해 상대적인 확률값을** 구한 값

<img width="1582" alt="5" src="https://github.com/junyong1111/junyong1111/assets/79856225/6bb25ef8-7d6d-486c-871d-555b0fbddba4">

## Transformer

<img width="1626" alt="6" src="https://github.com/junyong1111/junyong1111/assets/79856225/7d89ed45-1d09-4b71-aff6-a7981353753c">



### 1. 임베딩 과정

**기존의 임베딩 과정**

<img width="651" alt="7" src="https://github.com/junyong1111/junyong1111/assets/79856225/889e0ed6-d8e9-45c8-a84d-7432e108b5e6">

| I | embedding | embedding | embedding... |
| --- | --- | --- | --- |
| am | embedding | embedding | embedding... |
| a | embedding | embedding | embedding... |
| teacher | embedding | embedding | embedding... |
- 행 : 단어의 개수
- 열 : embedding dim (논문에서는 512)

**트랜스포머의 임베딩**

<img width="1381" alt="8" src="https://github.com/junyong1111/junyong1111/assets/79856225/104c8a70-296f-45bc-a9ee-311f2f619b23">



RNN은 순차적으로 입력값이 들어오지만 트랜스포머같은 경우 위치정보가 없으므로 **위치정보를 임베딩과정**에서 같이 넣어줘야 한다.

### 2. 인코더

<img width="1381" alt="9" src="https://github.com/junyong1111/junyong1111/assets/79856225/c115a936-d37a-4b2b-9c85-184494e9522c">

<img width="1263" alt="10" src="https://github.com/junyong1111/junyong1111/assets/79856225/ec9b0264-6ad7-404a-b3a1-aa53c5c158c4">


<img width="1559" alt="11" src="https://github.com/junyong1111/junyong1111/assets/79856225/d0f1e100-e833-4a58-a523-af160fd4ca0b">



위 과정에서 각각의 레이어는 독립된 파라메터를 사용한다.

### 3. 인코더 및 디코더 아키텍쳐

<img width="1617" alt="12" src="https://github.com/junyong1111/junyong1111/assets/79856225/b9185f6c-34a5-4d82-8a14-f640659a585a">

인코더의 가장 마지막 레이어의 출력값을 디코더에 넣어준다.

- 인코더의 어떤 부분에 집중해야하는지 하기 위해서 값을 넣어줌
- 마지막 레이어가 아닌 모든 레이어의 출력값을 받는 기법도 존재함

### 4. 디코더

<img width="788" alt="13" src="https://github.com/junyong1111/junyong1111/assets/79856225/5f361e23-c969-4c93-91f1-604923925535">


- 1개의 디코더 레이어에서는 2개의 어텐션을 사용
- 1번 어텐션
    - 인코더와 마찬가지로 각각의 단어들의 어텐션 스코어를 계산한다.
- 2번 어텐션(인코더 디코더 어텐션)
    - 인코더에 대한 정보를 어텐션하여 각각의 출력단어와 인코더의 출력정보에 대한 연관성을 확인

<img width="1485" alt="14" src="https://github.com/junyong1111/junyong1111/assets/79856225/cc083776-747f-4dfe-b917-99e7c0a6b045">


### Multi Head Attention

<img width="1583" alt="15" src="https://github.com/junyong1111/junyong1111/assets/79856225/27fb898f-9ad2-4264-b153-b2056dc9d724">


<img width="1611" alt="16" src="https://github.com/junyong1111/junyong1111/assets/79856225/5909c4b4-4fa2-4612-be8d-b685d68e4067">



**서로 다른 h개의 어텐션을 학습** 

<img width="1514" alt="17" src="https://github.com/junyong1111/junyong1111/assets/79856225/d6af664b-1a76-468d-bfee-60922ae9e776">


### 동작 과정

<img width="1403" alt="18" src="https://github.com/junyong1111/junyong1111/assets/79856225/d126aa56-a9c2-4446-b00d-4677e8e0de5d">

—# 현재 예제에서 차원은 4차원 헤드는 2개로 가정

<img width="1583" alt="19" src="https://github.com/junyong1111/junyong1111/assets/79856225/110f1740-82ed-469e-bdf0-447985fde279">


```python
I love you라는 단어가 들어온 경우 
1. I라는 단어는 I, love, you 각각의 키값과 곱해져서 어텐션값을 구한다.
2. 이후 소프트맥스를 통해 각각의 키값들에 대한 가중치를 구한다.
3. weight sum을 통해 최종 어텐션 스코어를 구한다.
```

**소스데이터로 I love you 라는 문장이 들어왔을 경우 아래와 같이 각 단어에 해당하는 embedding_dim * head 크기의 Query, Key, Value 행렬을 만들 수 있다.**

<img width="1590" alt="20" src="https://github.com/junyong1111/junyong1111/assets/79856225/4172ea91-5a09-4b65-a32e-13f9d2f6a8ec">



**위에서 만들어진 각각의 행렬을 이용하여 어텐션 스코어를 계산한다.**

- **Query * Key = AttentionScore**
    - **(단어의 갯수 x 단어의 갯수)**
- **AttentionScore = Softmax(AttentionScore) * Value**
    - (단어의 갯수 x head의 갯수)

<img width="1493" alt="21" src="https://github.com/junyong1111/junyong1111/assets/79856225/54603587-9521-475d-bd10-7ee97e846734">


**디코더에서는 마스크행렬을  이용하여 특정 값을 무시할 수 있음.**

<img width="1557" alt="22" src="https://github.com/junyong1111/junyong1111/assets/79856225/926371ca-5fed-4eda-93b3-693b4c5b7a8e">
**입력 차원과 같은 차원으로 만들 수 있음**

<img width="1573" alt="23" src="https://github.com/junyong1111/junyong1111/assets/79856225/e2ba31f7-09d4-45c4-8c96-24b257c01419">

<img width="1504" alt="24" src="https://github.com/junyong1111/junyong1111/assets/79856225/7b90ec0b-a873-445e-bf4a-45dc6c87ba6f">

### 어텐션 값을 구하는 방법(셀프 어텐션)

<img width="1552" alt="25" src="https://github.com/junyong1111/junyong1111/assets/79856225/8be10938-5337-42c8-ac7d-f1cc75b59dd9">



[[DMQA Open Seminar] Graph Attention Networks](https://www.youtube.com/watch?v=NSjpECvEf0Y)

[[딥러닝 기계 번역] Transformer: Attention Is All You Need (꼼꼼한 딥러닝 논문 리뷰와 코드 실습)](https://www.youtube.com/watch?v=AA621UofTUA)