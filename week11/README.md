- ML-GAT : A multi Level Graph Attention Model for Stock Prediction
- Meta Learning With Graph Attention Networks for low-Data Drug Discovery
- SGAT : Simplicial Graph Attention Network
- Graph attention-based collaborative filtering for user-specific recommender system using knowledge graph and deep neural networks
- Multi-Agent Trajectory Prediction With Heterogeneous Edge-Enhanced Graph Attention Network

[2207.11761.pdf](https://prod-files-secure.s3.us-west-2.amazonaws.com/60b7a12c-d798-4819-a350-0b02a52409b9/0e70bdfa-d8a4-4f55-b789-90bfece02cd0/2207.11761.pdf)

Light GCN은 유저-아이템의 2가지의 이종그래프인데 이를 확장한다면?

### **Abstract**

 이질적인 그래프는 여러 종류의 노드와 엣지를 가지고 있으며, 동질적인 그래프보다 **의미론적으로 풍부**하다. 이러한 복잡한 의미를 학습하기 위해, 많은 그래프 신경망 방법들이 이질적 그래프에서 노드 간의 다중 홉 상호작용을 포착하기 위한 메타패스를 사용한다. 일반적으로, 비대상 노드의 특성은 학습 과정에 포함되지 않지만, 여러 노드나 엣지를 포함하는 비선형 고차 상호작용이 존재할 수 있다. 본 논문에서는 이러한 고차 상호작용을 표현하기 위해 **Simplicial Graph Attention Network (SGAT)를 제안한다.**  SGAT는 비대상 노드의 특성을 단체에 배치하고, **Attention 메커니즘과** 상위 인접성을 사용하여 표현을 생성한다. 우리는 이질적인 그래프 데이터셋에서 노드 분류 작업을 통해 SGAT의 효과를 실증적으로 보여주고, 무작위 노드 특성을 사용하여 구조적 정보를 추출하는 SGAT의 능력을 추가로 보여준다. 수치 실험 결과, SGAT가 기존의 다른 최신 이질적 그래프 학습 방법들보다 더 나은 성능을 보임을 나타낸다.

**비대상 노드의 특성을 단체에 배치한다'는 것은**, 특정 사용자가 **평가하지 않은 영화의 정보도 다른 사용자들의 평가를 통해 간접적으로 학습에 포함**시키는 것을 의미한다.

예를 들어, 사용자 A와 사용자 B가 모두 '스타워즈'를 평가했고, 사용자 B와 사용자 C가 '인디아나 존스'를 평가했다고 하면, 사용자 A는 '인디아나 존스'를 평가하지 않았지만, SGAT는 사용자 B를 통해 사용자 A와 '인디아나 존스' 사이의 간접적인 연결을 학습할 수 있다. 여기서 '스타워즈'와 '인디아나 존스', 그리고 그 영화들을 평가한 사용자들 간의 관계는 단체(simplicial complex)를 이루어, 모델이 이 데이터를 기반으로 더 풍부한 사용자의 취향 프로파일을 만들 수 있도록 도와준다.

이렇게 SGAT는 단순히 개별 평가 점수를 넘어서, 사용자들의 영화 평가 패턴과 그들이 어떻게 서로 관련되어 있는지를 포괄적으로 파악하여 추천 시스템의 정확도를 향상시킬 수 있다.

### 1. **Introduction**

초기에 동질적 그래프에 초점을 맞춘 모델로부터 실세계 데이터를 더 잘 대표할 수 있는 이질적 그래프를 처리할 수 있는 더 발전된 접근법으로의 전환을 설명한다.

논문은 현재 이질적 그래프에 사용되는 **메타패스 기반 방법의 한계**를 지적한다. 이에는 메타패스 선택에 대한 민감성, 비대상 노드 기능의 누락, 그리고 단순한 쌍별 상호작용만으로는 복잡한 관계를 포착하는 데 불충분함이 포함된다. 이러한 문제를 해결하기 위해, 논문은 고차원 관계를 모델링하기 위해 단체 복합체를 사용하고 효과적인 임베딩을 학습하기 위해 상위 인접성과 함께 주의 메커니즘을 사용하는 Simplicial Graph Attention Network (SGAT)를 소개한다.

SGAT는 비대상 노드의 특성을 포함하도록 설계되었고, 이질적 그래프를 동질적 단체로 변환하는 동안 잠재적 정보의 손실을 피하면서 각 오더의 단체의 중요성을 학습할 수 있다. 서론은 논문의 주요 기여를 요약하여 마무리하는데, 이에는 이질적 그래프로부터 단체 복합체를 구성하는 방법, 단체 복합체에 맞춤화된 GAT-류의 주의 메커니즘, 그리고 현재 최신 모델들과 비교하여 이질적 그래프 데이터셋에서 SGAT의 노드 분류 작업의 우수한 성능을 실증적으로 보여주는 것이 포함된다.

예를 들어, 학술 네트워크에서 "저자(Author) - 작성(Paper) - 주제(Topic)"라는 메타패스는 저자가 작성한 논문이 특정 주제에 속함을 나타내는 경로를 정의한다. 이런 메타패스를 사용하면 저자의 연구 분야를 파악하거나, 주제별로 연구자들을 그룹화하는 등의 분석이 가능해진다.

메타패스 기반 방법은 이렇게 정의된 메타패스를 통해 이질적 그래프에서 유의미한 정보를 추출하고, 그래프 내에서의 복잡한 상호작용을 이해하는 데 도움을 준다. 하지만, 메타패스를 잘못 선택하거나, 중요한 비대상 노드의 정보를 누락하는 등의 문제가 발생할 수 있다. 이러한 문제를 해결하기 위해 최신 연구에서는 메타패스 선택에 대한 의존도를 줄이고, 그래프의 모든 정보를 효과적으로 활용할 수 있는 새로운 방법론을 모색하고 있다.

### **2. Related Work**

 이종 그래프 학습은 주로 사전 정의된 메타패스를 사용하거나 훈련 중 최적의 메타패스를 학습하는 것을 포함한다. 이 분야에서 주목할 만한 작업으로는 **HAN, HERec, metapath2vec, Graph Transformer Network (GTN), REGATHER** 등이 있다.

HAN은 이종 그래프를 메타패스 기반 동질적 그래프로 변환하고 이중 레벨 Attention를 적용하지만, 비대상 노드의 특징은 고려하지 않습니다. HERec는 메타패스를 사용하여 무작위 걸음을 생성하고, 시작 노드 유형에 초점을 맞추어 다른 노드 유형을 같은 공간에 표현하지 않다. Metapath2vec는 메타패스를 사용하여 의미 관계를 포함하고 유형 편향을 줄이는 무작위 걸음을 유도한다. GTN은 후보 인접성을 결합하여 메타패스 기반 구조를 학습하는 반면, REGATHER는 메타패스 기반 구조를 포함하는 멀티홉 관계 유형의 서브그래프를 생성하고 주의 메커니즘을 적용하여 서브그래프에 가중치를 할당한다.

또한 M-HIN, mg2vec, Meta-GNN과 같은 메타그래프 기반 방법을 언급하는데, 이 방법들은 더 세밀하고 비선형적인 의미를 포착하고자 한다. 이러한 방법들은 노드 표현의 학습을 안내하고 노드 간의 잠재적 관계를 포착하기 위해 메타그래프를 사용한다.

동질적 GNN 접근 방식으로는 Message Passing Simplicial Network (MPSN) 및 Simplicial Neural Network (SNN)과 같이 단체 복합체를 사용하는 것이 있다. MPSN은 단체 복합체에 대한 일반적인 메시지 전달 프레임워크를 도입하는 반면, SNN은 이를 이질적 이분 그래프의 맥락에서 사용한다. 그러나 SNN은 이분 그래프에 한정되어 있으며 누락된 데이터를 채우는 데 중점을 둔다.

| 모델/기법 | 설명 | 특징 |
| --- | --- | --- |
| HAN (Heterogeneous Graph Attention Network) | 이질적 그래프를 메타패스 기반 동질적 그래프로 변환 | 비대상 노드의 특성을 고려하지 않음 |
| HERec | 메타패스 기반 무작위 걸음을 생성 | 시작 노드 유형만을 고려하여 다른 노드 유형과 혼합을 피함 |
| metapath2vec | 메타패스를 사용하여 의미론적 관계를 포함하는 무작위 걸음을 안내 | 노드 유형에 대한 편향 감소에 초점 |
| GTN (Graph Transformer Network) | 후보 인접성을 결합하여 메타패스 기반 구조를 학습 | 메타패스 선택의 어려움을 해결하기 위한 시도 |
| REGATHER | 멀티홉 관계 유형의 서브그래프 생성 | 메타패스 기반 구조를 간접적으로 포함하고 주의 메커니즘을 통해 가중치 할당 |
| M-HIN | 메타그래프를 사용하여 노드 간의 관계를 캡처 | 복잡한 임베딩 체계를 사용하여 더 정확한 노드 특성 포착 |
| mg2vec | 메타그래프와 노드의 임베딩을 동시에 학습 | 노드 표현의 학습을 안내하고 노드 간 잠재적 관계 포착 |
| Meta-GNN | 메타그래프 컨볼루션 레이어를 도입 | 노드의 수용 필드를 정의하고 다른 메타그래프에서 노드 특성을 결합하기 위해 주의 메커니즘 사용 |
| MPSN (Message Passing Simplicial Network) | 단체 복합체에 대한 일반적인 메시지 전달 프레임워크 도입 | 동질적 그래프에 한정되며, 이질적 그래프에 대한 적용 미탐색 |
| SNN (Simplicial Neural Network) | 이질적 이분 그래프에 단체 복합체 사용 | 이분 그래프에 한정되고, 누락된 데이터 채우기에 중점 |

### 3. Simplicial Graph Attention Network

SGAT(Simplicial Graph Attention Network) 섹션은 SGAT가 어떻게 여러 단체 복합체(simplicial complexes)를 통합하여 k차원 단체들을 설명하는 데 있어 다양한 복합체의 중요성을 계정하는지에 대한 메커니즘을 다룬다. 이 과정에는 주어진 범위 내에서 각 η에 대해 k-단체들을 생성하고, 각 특정 단체 복합체에 대한 임베딩을 생산하기 위해 단체 주의 계층(simplicial attention layer)을 적용하는 것이 포함된다. SGAT 프레임워크는 η와 독립적인 각 단체에 대한 고유한 k-튜플을 포함하며, η 그룹의 k-단체 임베딩을 결합하기 위해 주의 계층을 적용한다.

임베딩은 배운 가중치를 통해 모든 η 값에 걸쳐 통합되어 각 단체에 대한 최종 임베딩을 생성한다. 이 방법은 각각의 k차원 단체를 표현하는데 다양한 단체 복합체의 중요성을 고려한다. SGAT 아키텍처는 메타패스나 메타그래프에 의존하지 않으므로, 메타패스 선택에 대한 민감성, 비대상 노드 특성의 누락, 노드 간 복잡한 상호작용을 포착하는 데 있어 제한된 표현력과 같은 그들의 전형적인 제한을 피한다.

SGAT는 다중 대상 노드 사이의 고차 관계를 포착하기 위해 단체 복합체 개념을 사용하며, 변환 과정 동안 잠재적 정보의 손실을 방지하기 위해 단체에 비대상 노드의 특성을 포함한다. SGAT의 변형은 성능을 향상시키기 위해 엣지 특성을 포함한다.

수치 실험은 SGAT와 그 엣지 특성 변형(SGAT-EF)이 이질적 그래프에서 노드 분류 작업에 있어 여러 최신 모델을 능가함을 보여준다. 또한, SGAT는 임의의 노드 특성을 사용할 때 분명한 이점을 보여주어, 오직 원래 노드 특성에만 의존하지 않고 그래프의 구조적 정보를 효과적으로 추출하고 활용할 수 있음을 나타낸다.

1. **단체 복합체 생성**: 사용자, 영화, 평점, 태그 등의 노드 유형에 대한 단체 복합체를 생성한다. 예를 들어, 한 사용자가 여러 영화에 평점을 준 경우, 이러한 관계를 나타내는 2-단체(삼중 관계)를 만들 수 있다.
2. **임베딩 생성**: SGAT의 단체 주의 계층을 사용하여 각 단체 복합체에 대한 임베딩을 생성한다. 이는 각 노드의 특성과 이들이 형성하는 복잡한 관계를 함께 고려하는 과정이다.
3. Attention **메커니즘을 통한 통합**: 생성된 임베딩들을 통합하기 위해 Attention 메커니즘을 적용하여, 각각의 단체 복합체가 전체 그래프를 설명하는 데 얼마나 중요한지를 결정한다. 이는 다양한 관계의 중요성을 학습하는 과정이다.
4. **최종 임베딩**: 모든 임베딩들이 통합되어 각 노드에 대한 최종 임베딩이 만들어진다. 이 임베딩은 사용자가 영화를 어떻게 평가하는지, 태그와 영화의 관계는 어떠한지 등 복잡한 상호작용을 포착하는 데 사용된다.
5. **노드 분류**: 마지막으로, 생성된 최종 임베딩은 노드 분류 작업, 예를 들어 사용자의 취향 예측, 영화의 장르 분류 등에 사용된다.
- MovieLesn
    1. **데이터 전처리**: MovieLens 데이터셋에는 사용자(User), 영화(Movie), 평점(Rating), 태그(Tag) 등의 다양한 요소가 포함되어 있습니다. 우선, 이 요소들을 노드로 가정하고 사용자가 영화에 평점을 주거나 태그를 붙인 행동을 엣지로 표현합니다. 이렇게 구성된 데이터는 이질적 그래프로 변환됩니다.
    2. **단체 복합체 구축**: 이 데이터를 바탕으로 단체 복합체를 구축합니다. 예를 들어, 한 사용자가 특정 영화에 평점을 주고 태그를 달았다면, 이를 2-단체로 볼 수 있습니다. 여기서 사용자는 0-단체, 사용자와 영화의 평점 관계는 1-단체, 사용자-영화-태그의 관계는 2-단체로 표현될 수 있습니다.
    3. **임베딩 생성**: 각 단체 복합체에 대해 SGAT의 단체 주의 계층을 적용하여 임베딩을 생성합니다. 이 과정에서는 각 단체의 노드 특성과 복잡한 상호작용을 함께 고려하여 임베딩을 구성합니다. 예를 들어, 사용자와 영화, 태그 간의 상호작용을 반영하는 임베딩이 만들어집니다.
    4. **주의 메커니즘을 통한 통합**: 다음으로, 생성된 각 임베딩에 대해 주의 메커니즘을 적용하여 서로 다른 단체 복합체의 중요성을 평가하고, 이를 바탕으로 최종 임베딩을 생성합니다. 예를 들어, 평점이 높은 영화에 대한 사용자의 임베딩은 더 높은 가중치를 받을 수 있습니다.
    5. **최종 임베딩 생성**: 모든 η에 대한 임베딩이 합쳐져 각 노드에 대한 최종적인 임베딩이 생성됩니다. 이 최종 임베딩은 추후 영화 추천이나 사용자의 선호 분석 등에 사용될 수 있습니다.
    6. **노드 분류 작업**: 생성된 최종 임베딩을 사용하여, 예를 들어 사용자가 좋아할 만한 영화를 추천하거나, 영화의 장르를 분류하는 노드 분류 작업을 수행합니다. SGAT의 장점은 이러한 복잡한 상호작용을 잘 포착하여 더 정확한 추천을 가능하게 한다는 점입니다.
    7. **성능 평가**: SGAT의 성능을 평가하기 위해, 실제 사용자의 평점이나 영화 장르와 같은 실제 데이터와 SGAT가 예측한 결과를 비교합니다. 이때 SGAT가 메타패스 기반 모델들과 비교하여 얼마나 향상된 성능을 보이는지를 측정할 수 있습니다.
    
    <aside>
    💡
    
    1. **영화와 사용자 모으기**:
        - MovieLens에는 많은 사용자들이 여러 영화에 평점을 매긴 정보가 있습니다. 예를 들어, 사용자 A가 '어벤져스'와 '타이타닉' 영화에 평점을 줬다고 해봅시다.
    2. **관계 맺기**:
        - 사용자 A가 준 평점을 바탕으로 '어벤져스'와 '타이타닉' 영화 사이의 관계를 만듭니다. 이 관계를 우리는 '단체'라고 부릅니다.
    3. **중요한 관계 찾기**:
        - 모든 영화와 사용자 사이의 관계를 보고, 어떤 관계가 가장 중요한지를 결정합니다. 예를 들어, 사용자 A가 액션 영화에 높은 평점을 많이 주었다면, 액션 영화 관계를 더 중요하게 생각할 수 있습니다.
    4. **영화 추천하기**:
        - 위에서 찾은 중요한 관계를 바탕으로, 사용자 A가 좋아할 만한 새로운 영화를 추천합니다. 이 과정에서 SGAT는 사용자 A가 '어벤져스'와 '타이타닉'에 준 평점뿐만 아니라 다른 영화에 대한 평점 패턴도 고려합니다.
    
    SGAT는 복잡한 영화와 사용자 사이의 관계를 이해하고, 사용자에게 적합한 영화를 추천하는 데 도움을 주는 방법입니다. 이 모델은 단순히 영화가 장르에 속한다는 것보다 더 많은 정보를 분석하여 사용자의 취향을 더 정확하게 파악할 수 있습니다.
    
    </aside>
    

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader

# 데이터셋 로딩
# 데이터셋은 'userId', 'movieId', 'rating' 컬럼을 포함해야 합니다.
ratings = pd.read_csv('ratings.csv')

# 사용자와 영화의 ID를 숫자로 인코딩
user_encoder = LabelEncoder()
movie_encoder = LabelEncoder()

ratings['userId'] = user_encoder.fit_transform(ratings['userId'])
ratings['movieId'] = movie_encoder.fit_transform(ratings['movieId'])

# 단체 복합체를 위한 인접 행렬 생성 (간단화를 위해 평점 행렬을 사용)
num_users = ratings['userId'].nunique()
num_movies = ratings['movieId'].nunique()

# 사용자-영화 인접 행렬
user_movie_adj = torch.zeros(num_users, num_movies)
for _, row in ratings.iterrows():
    user_movie_adj[int(row['userId']), int(row['movieId'])] = row['rating']

# SGAT 모델 정의
class SGAT(nn.Module):
    def __init__(self, num_users, num_movies, embed_dim):
        super(SGAT, self).__init__()
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.movie_embed = nn.Embedding(num_movies, embed_dim)
        self.attention = nn.Parameter(torch.ones(embed_dim, 1))
        
    def forward(self, user_movie_adj):
        # 사용자와 영화의 임베딩
        user_embedding = self.user_embed.weight
        movie_embedding = self.movie_embed.weight
        
        # 임베딩 행렬의 곱
        energy = torch.matmul(user_embedding, movie_embedding.T)
        
        # 주의 메커니즘을 사용한 가중치 계산
        attention_weights = F.softmax(torch.matmul(energy, self.attention), dim=1)
        
        # 가중치를 사용한 임베딩 행렬의 업데이트
        user_embedding = torch.matmul(attention_weights.T, user_embedding)
        movie_embedding = torch.matmul(attention_weights, movie_embedding)
        
        return user_embedding, movie_embedding

# 모델 인스턴스화 및 옵티마이저 설정
embed_dim = 64
model = SGAT(num_users, num_movies, embed_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 훈련 루프 (간단화된 버전)
for epoch in range(5):
    model.train()
    optimizer.zero_grad()
    
    # 모델 예측
    user_embedding, movie_embedding = model(user_movie_adj)
    
    # 손실 계산 (여기서는 MSE를 사용)
    predicted_ratings = torch.matmul(user_embedding, movie_embedding.T)
    loss = F.mse_loss(predicted_ratings, user_movie_adj)
    
    # 역전파 및 옵티마이저 단계
    loss.backward()
    optimizer.step()
    
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# 모델을 사용하여 예측 수행
model.eval()
with torch.no_grad():
    user_embedding, movie_embedding = model(user_movie_adj)
    predicted_ratings = torch.matmul(user_embedding, movie_embedding.T)
    
# 예측된 평점을 실제 평점과 비교 (예시)
user_id = 0  # 예시 사용자 ID
predicted_ratings_for_user = predicted_ratings[user_id]
print(predicted_ratings_for_user)
```

# Attention-Based Recommendation On Graphs

[2201.05499.pdf](https://prod-files-secure.s3.us-west-2.amazonaws.com/60b7a12c-d798-4819-a350-0b02a52409b9/1273b1b7-2141-45e3-b9c7-761d62b0c79f/2201.05499.pdf)

### ABSTRACT

그래프 신경망(Graph Neural Networks, GNN)은 다양한 작업에서 뛰어난 성능을 보여왔지만, 추천 시스템에서 GNN을 연구한 사례는 많지 않다. 이 연구에서 제안된 **GARec**은 사용자와 아이템의 임베딩을 추출하기 위해 **Attention 메커니즘**을 사용하여 공간적 그래프 합성곱 네트워크(Graph Convolutional Network, **GCN**)를 추천 그래프에 적용하는 모델 기반 추천 시스템이다. Attention 기반 메커니즘이 GCN에게 관련 사용자 또는 아이템이 대상 엔티티의 최종 표현에 얼마나 영향을 줄지를 알려준다. GARec의 성능은 RMSE(Root Mean Square Error, 평균 제곱근 오차) 측면에서 기존의 모델 기반 및 그래프 신경망 방법들과 비교되었고, 다양한 MovieLens 데이터셋에서 기존 방법들보다 우수한 성능을 나타냈다.

### 1. Introduction

추천 시스템 분야에서 그래프 기반 모델링은 엔티티 간의 간접적인 관계까지 포함하는 유사성 개념을 확장하여 데이터의 희소성 문제를 극복하는 데 도움이 된다. 그래프에서 실제 유사성을 반영할 수 있는 신뢰할 수 있는 및 불신뢰할 경로를 인식하는 방법은 기존 방법론의 주요 도전 과제이다.

이 논문은 직접적인 유사성 계산을 모델 기반 접근법으로 대체하고 사용자 및 아이템 노드를 임베딩하여 추천을 추출하는 심층 학습 기반 방법론에 대한 증가하는 관심을 언급한다. GNN은 그래프 내의 노드를 위한 임베딩을 추출하는 일반적인 접근법으로, 사용자와 아이템 노드를 포함한 추천 그래프에 적용될 수 있다.

이 논문은 사용자와 아이템 노드를 임베딩하기 위한 새로운 그래프 신경망을 제안하며, 다양한 분야에서 임베딩 추출에 효과적임이 입증된 **Attention 메커니즘**을 활용한다. Attention 기반 메커니즘은 대상 노드의 임베딩을 추출하는 동안 더 많은 정보를 제공하는 관련 노드를 인식하고 사용한다.

제안하는 방법은 **유사성 기반 방법과 모델 기반 접근법**의 이점을 모두 활용한다. 이는 대상 노드의 임베딩을 추출할 때 이웃 정보를 통합하고, 관련 사용자와 공동 평가 관계가 있는 사용자의 관련성을 고려한다. 그러나 수동으로 특성을 추출할 필요 없이 심층 신경망 모델을 사용하여 추천을 추출하는 방식으로, 이웃 정보를 활용하면서 오프라인 훈련 및 온라인 추천을 지원한다.

본 논문은 기존 GNN에 적용된 Attention 기반 메커니즘의 결함을 수정하는 방법을 제안하며, 그래프의 모든 이웃에게 동일한 주의 점수를 부여하는 기존 GAT(Graph Attention Network)의 단점을 해결한다. 본 연구는 그래프에서 노드의 표현을 학습하기 위해 그래프 신경망과 행렬 분해 방법을 결합한 하이브리드 방식을 제시하고, 표준 Attention 메커니즘의 성능을 향상시키며, 최신 기준을 뛰어넘는 새로운 모델 기반 협업 필터링 프레임워크를 소개한다.

### 2. Background

 GNN은 그래프의 노드 정보와 위상 구조로부터 학습하는 딥러닝 모델이다. 처음에는 단순한 신경망을 사용하여 노드와 엣지의 레이블 정보를 통합하여 각 노드의 표현을 생성하는 데 사용되었다. 이후 다양한 분야에서 노드 분류, 클러스터링, 링크 예측, 그래프 분류 및 시각화에 적용되었다.

GNN은 주로 **집계(aggregate) 및 갱신(updater) 모듈**을 사용하여 노드를 임베딩하는데, 집계 모듈은 이웃 노드로부터 특징 정보를 모으고, 갱신 모듈은 모아진 정보를 사용하여 대상 노드의 임베딩을 업데이트한다. 여기서 중요한 결정은 **얼마나 많은 정보가 노드에서 그 이웃으로 전파되어야 하는지, 그리고 노드의 이웃으로부터 받은 정보를 어떻게 집계할 것**인지다.

Graph Convolutional Neural Network (GCN)는 가장 인기 있고 자주 사용되는 GNN 방법 중 하나이다. 스펙트럼-GCN은 그래프를 라플라시안 행렬에 기반하여 표현하는 반면, 공간-GCN은 다양한 이웃 크기를 처리하고 노드에 대한 가중치 행렬을 정의하는 데 어려움이 있다. Graph Attention Network (GAT)는 GNN의 전파 단계에서 주의 메커니즘을 채택한 공간-GCN 방법입니다. 이는 각 이웃에 대한 주의 계수를 계산하여 대상 노드에 대한 상대적인 유사성에 기반하여 정보를 다르게 집계할 수 있도록 한다.

요약하자면, "Background" 섹션은 GNN의 기본적인 개념과 구조, 그리고 GCN 및 GAT와 같은 특정 GNN 유형에 대해 소개하고 설명한다. 이는 노드 임베딩을 생성하고, 이웃으로부터 정보를 집계하는 과정에 대한 이해를 제공하며, 이후 섹션에서 설명되는 GARec 모델의 기초를 마련한다.

### 3. Related Work

![스크린샷 2023-11-14 오후 2.01.13.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/60b7a12c-d798-4819-a350-0b02a52409b9/8470b0be-448c-4b19-bbf5-16e3541fbf33/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2023-11-14_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_2.01.13.png)

이 섹션은 GNN이 추천 시스템에서 어떻게 유용한 정보를 제공하는지, 그리고 이전의 연구와 비교하여 GARec이 어떤 혁신을 제공하는지를 설명한다.

여기서는 사용자와 아이템 간의 다중 홉 연결을 모델링함으로써 협업 필터링의 성능을 향상시킬 수 있는 여러 연구를 검토한다. 여러 방법론이 소개되는데, 이는 다양한 유형의 그래프를 모델링하고 다른 알고리즘을 통해 최종 추천을 도출하는 방식에서 주로 차이가 난다.

이 섹션에서 언급된 주요 연구들은 다음과 같습니다:

1. MGCNN: 사용자 피드백과 함께 부가 정보를 통합하여 추천을 도출
2. GC-MC: 그래프 오토인코더를 사용하여 사용자와 아이템 노드 간의 엣지의 가중치를 직접 예측
3. PinSage: 대규모 데이터셋에서 GCN을 사용하여 임베딩을 처리
4. GraphRec: 소셜 네트워크의 이웃 사용자와 상호작용한 아이템으로 사용자를 표현
5. NGCF: 사용자와 아이템의 고차 간접 연결성을 고려
6. IG-MC: 사용자-아이템 서브그래프에 그래프 레벨 GCN을 적용

GARec 모델은 이러한 관련 연구를 기반으로 하여 **어텐션 메커니즘**을 사용하여 사용자와 아이템의 임베딩을 개선한다. 이는 각 이웃에 대한 주의 계수를 계산하고, 해당 계수를 사용하여 대상 노드의 임베딩을 개선하는 데 중점을 둔다. 이를 통해 최종적으로 사용자와 아이템 간의 관계를 예측하는 성능이 향상된다.

**GARec은 어텐션 기반 메커니즘을 통한 임베딩의 품질 향상과 함께,  사용자의 가능한 관심사를 예측하고 추천 리스트를 생성하는 데 있어서 기존 모델들보다 뛰어난 성능을 제공한다.**

### 4. Method

GARec, 즉 Attention 기반 그래프 추천 시스템의 세부 방법론에 대해 설명한다.

GARec은 사용자와 아이템 간의 관계를 예측하기 위해 그래프 컨볼루셔널 네트워크(**GCN**)와 **Attention** 메커니즘을 활용한 모델 기반 **협업 필터링 방식**이다. 이 모델은 사용자-아이템 상호작용 행렬과 초기 특성 벡터를 입력으로 받아 사용자와 아이템의 임베딩을 동시에 학습한다.

GARec의 중요한 특징은 Attention 메커니즘을 통해 중요한 이웃 정보를 효과적으로 활용하여 각 노드의 임베딩을 추출한다는 점이다. 이를 위해 모델은 사용자와 아이템 각각에 대해 직접적이거나 간접적으로 연결된 이웃을 식별하고, 이들 이웃의 정보를 기반으로 해당 노드의 임베딩에 얼마나 기여할지 결정하는 Attnetion 계수(attention coefficient)를 계산한다.

모델 아키텍처는 다음과 같은 단계로 구성된다.

1. **문제 설정(Problem Setup)**: 사용자-아이템 상호작용을 나타내는 그래프 구조를 설정
2. **모델 아키텍처(Model Architecture)**: 사용자와 아이템 간의 관계를 예측하기 위해 GARec의 구조를 설계한다.
3. **이웃 선택(Neighbor Selection)**: 사용자와 아이템의 임베딩을 대표하기 위해 중요한 이웃을 선택한다.
4. **초기 특성 벡터(Initial Feature Vector)**: 사용자와 아이템의 정보를 벡터 형태로 변환이를 위해 행렬 분해 방식을 사용한다.
5. **집계기(Aggregator)**: 각 노드로부터 이웃의 정보를 집계한다. 주의 계수를 사용하여 이웃의 정보를 가중 평균한다.
6. **업데이터(Updater)**: 집계된 정보를 기반으로 노드의 임베딩을 업데이트한다.
7. **평가 예측(Rating Prediction)**: 사용자와 아이템의 임베딩을 기반으로 평가를 예측한다.

GARec는 기존의 Attention 메커니즘을 수정하여 더 나은 표현을 생성하고 성능을 향상시킨다. 이 모델은 사용자와 아이템 간의 상호작용을 예측하는데 필요한 임베딩을 생성하고, 그래프 구조와 Attention 기반 메커니즘을 사용하여 사용자의 임베딩 과정에서 중요한 이웃의 정보를 더 많이 고려한다.

![스크린샷 2023-11-14 오후 2.18.25.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/60b7a12c-d798-4819-a350-0b02a52409b9/35b782fd-12b2-450f-a74d-6c4b20167e60/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2023-11-14_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_2.18.25.png)

### 5. Experimental Studies and Results

 GARec 알고리즘의 성능을 평가하기 위해 설계된 실험들과 그 결과에 대해 설명한다.

실험은 MovieLens 100K와 MovieLens 1M 데이터셋을 사용하여 수행되었다. 이 데이터셋들은 사용자의 명시적 피드백을 1에서 5까지의 등급으로 나타낸 것이며, 추가적으로 사용자의 연령, 성별, 직업 등과 영화의 장르, 출시 날짜 같은 부가 정보를 포함한다. 실험 설정은 80/20과 90/10의 훈련/테스트 분할 비율을 적용하고, 5겹 교차 검증을 사용한다.

GARec은 여러 기준 모델들과 비교되었으며, 이 기준 모델들에는 행렬 분해 방법(NMF, PMF, SVD++), 그래프 신경망(GC-MC, PinSage, IGMC), 그리고 신경망 모델(AutoRec, Wide & Deep, DMF) 등이 포함되었다. 평가 척도로는 회귀 작업에 널리 사용되는 RMSE(평균 제곱근 오차)가 사용되었다.

실험 결과 GARec는 RMSE 면에서 다른 모든 기준 모델들보다 낮은 값을 기록하며 더 좋은 성능을 보여주었다. 특히, MovieLens 100K 데이터셋에서는 GARec가 SVD++보다 0.7% 더 낮은 RMSE를, AutoRec보다는 6%, 그리고 GC-MC보다는 2% 더 낮은 RMSE를 달성했다.

이 결과는 GARec의 장점을 보여준다. GARec는 그래프 기반 모델링과 신경망의 비선형성을 결합하여, 엔티티 간의 간접적인 관계를 파악하고 이를 목표 변수에 매핑하기 전에 해당 엔티티로부터 정보를 집계할 수 있다. 또한, 그래프 내에서 가치 있는 정보를 추출하여 GARec가 그래프가 없는 방법들보다 더 나은 성능을 보이게 한다. GARec의 MLP 부분은 다른 GNN 방법들의 예측 부분과 유사하지만, 임베딩의 품질이 전체 알고리즘 성능을 향상시키는 핵심 요소이다.

```python
import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class LightGCNLayer(MessagePassing):
    def forward(self, x, edge_index):
        # Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Start propagating messages.
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # Normalize node features.
        return norm.view(-1, 1) * x_j

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, num_layers):
        super(LightGCN, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.layers = nn.ModuleList([LightGCNLayer() for _ in range(num_layers)])

        # Initialize weights.
        self.user_embedding.weight.data.normal_(0, 0.1)
        self.item_embedding.weight.data.normal_(0, 0.1)

    def forward(self, user_indices, item_indices, edge_index):
        # Initial user and item embeddings.
        u_embeddings = self.user_embedding(user_indices)
        i_embeddings = self.item_embedding(item_indices)

        all_embeddings = torch.cat([u_embeddings, i_embeddings], dim=0)

        embs = [all_embeddings]
        for layer in self.layers:
            all_embeddings = layer(all_embeddings, edge_index)
            embs.append(all_embeddings)

        embs = torch.stack(embs, dim=1)
        lightgcn_out = torch.mean(embs, dim=1)

        users, items = torch.split(lightgcn_out, [num_users, num_items], 0)
        users = users[user_indices]
        items = items[item_indices]

        # Compute prediction scores.
        scores = (users * items).sum(dim=1)
        return scores

    def recommend(self, user_indices, edge_index):
        # Compute user embeddings.
        u_embeddings = self.user_embedding(user_indices)

        all_embeddings = torch.cat([u_embeddings, self.item_embedding.weight], dim=0)

        for layer in self.layers:
            all_embeddings = layer(all_embeddings, edge_index)

        users, items = torch.split(all_embeddings, [u_embeddings.size(0), self.item_embedding.weight.size(0)], 0)
        scores = torch.matmul(users, items.t())

        return scores
```

- 100개로 늘려보기
    - 점점 확장(100단위)
    - 데이터셋을 큰 데이터를 사용
- Attention 5%내라도 개선이 된다면

