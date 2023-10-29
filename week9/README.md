# Light GCN  개선 방향

1. **Attention Mechanism**:
    - **기본 개념**: Attention 메커니즘은 모델이 주요한 정보에 더 많은 집중을 할 수 있도록 하는 기법이다.
    - **적용 방법**: Graph Attention Network (GAT)는 그래프 기반 모델에 attention 메커니즘을 적용한 대표적인 방법이다. 각 edge의 가중치는 연결된 노드들의 특징에 기반하여 동적으로 계산된다.
    - **장점**: 모든 관계가 동일하게 중요한 것이 아니기 때문에, attention을 이용하면 특정 관계에 집중함으로써 모델의 성능을 향상시킬 수 있다.
2. **Temporal Dynamics**:
    - **기본 개념**: 사용자와 아이템 간의 상호작용은 시간에 따라 변화한다. 이러한 시간적 동적을 모델에 포함시키는 것이 필요하다.
    - **적용 방법**: Temporal Graph Networks (TGN)와 같은 모델들은 그래프의 시간적 동적을 처리한다. 이러한 모델은 각 노드와 edge에 시간적인 정보를 포함시켜 시간에 따른 상호작용의 변화를 학습한다.
    - **장점**: 시간의 흐름에 따른 사용자의 선호도 변화나 아이템의 인기도 변화를 반영하여, 더욱 정확한 추천을 제공한다.
3. **Hybrid Model**:
    - **기본 개념**: 여러 추천 모델의 장점을 결합하여 하나의 모델을 구성하는 것이다.
    - **적용 방법**: 예를 들면, LightGCN과 matrix factorization 기법을 함께 사용하는 것이다. Matrix factorization은 사용자-아이템 상호작용 행렬의 빈칸을 채워넣는 데 효과적이지만, 복잡한 사용자-아이템 관계를 학습하는 데에는 한계가 있다. 반면, LightGCN은 그래프 구조를 활용하여 복잡한 관계를 학습한다. 이 두 방법을 결합하면, 각각의 장점을 최대한 활용할 수 있다.
    - **장점**: 여러 모델의 장점을 결합함으로써, 단일 모델로는 얻을 수 없는 성능 향상을 기대할 수 있다.

### **Hybrid Model (MF + Light GCN)**

**MovieLens 데이터에 맞는 MF 모델 정의**

- #— 데이터 전처리 코드를 통해 반환되는 데이터는 다음과 같다.
    
    `return tet_edge_data, tet_sparse_data, user_mapping, item_mapping`
    
    ```python
    #--  [edge_index, train_edge_index, val_edge_index, test_edge_index]
    print(type(tet_edge_data))
     
    print(tet_edge_data[0]) #-- 전체 index
    print(tet_edge_data[1]) #-- train
    print(tet_edge_data[2]) #- val 
    print(tet_edge_data[3]) #-- test
    
    """
    <class 'list'>
    tensor([[   0,    0,    0,  ...,  609,  609,  609],
            [   0,    2,    5,  ..., 9461, 9462, 9463]])
    tensor([[  60,  124,  409,  ...,   65,  163,  427],
            [1002, 4705,  820,  ..., 6422, 1044, 5309]])
    tensor([[ 413,  316,   52,  ...,  325,  592,  481],
            [4911, 8537,  357,  ..., 5843, 2546, 5269]])
    tensor([[ 549,    6,  413,  ...,  203,  233,  476],
            [7693, 2355,  483,  ..., 7456,  958, 1883]])
    """
    
    print(tet_edge_data[1][0]) #-- train
    print(tet_edge_data[1][1]) #-- train
    """
    tensor([ 60, 124, 409,  ...,  65, 163, 427])
    tensor([1002, 4705,  820,  ..., 6422, 1044, 5309])
    """
    ```
    
    `blog_tet_edge_data[1][0]`는 사용자 노드의 인덱스를,
    
    `blog_tet_edge_data[1][1]`는 아이템 노드의 인덱스를 나타내는 텐서(tensor)이다.
    
    예를 들어, 출력값의 첫 번째 엔트리를 보면:
    
    - `blog_tet_edge_data[1][0][0]`의 값은 60
    - `blog_tet_edge_data[1][1][0]`의 값은 1002
    
    이는 60번 사용자가 1002번 아이템에 대한 상호작용(예: 레이팅, 구매, 클릭 등)을 가지고 있다는 것을 의미한다.
    
- 모델 정의
    
    ```python
    class MatrixFactorization(nn.Module):
        def __init__(self, num_users, num_items, embedding_dim):
            super(MatrixFactorization, self).__init__()
            
            self.user_embedding = nn.Embedding(num_users, embedding_dim)
            self.item_embedding = nn.Embedding(num_items, embedding_dim)
            
            # 임베딩 초기화
            self.mf_user_embeddings = nn.Embedding(num_users, embedding_dim)
            self.mf_item_embeddings = nn.Embedding(num_items, embedding_dim)
    		    
    				nn.init.normal_(mf_user_embeddings.weight, std=0.1)
    				nn.init.normal_(mf_item_embeddings.weight, std=0.1)
    
        def forward(self, user_indices, item_indices):
            user_embedding = self.user_embedding(user_indices)
            item_embedding = self.item_embedding(item_indices)
            
            # 내적 연산으로 평점 예측
            predictions = (user_embedding * item_embedding).sum(1)
            return predictions
    ```
    
- 모델 디테일 확인
    
    ```python
    num_users = blog_tet_edge_data[0][0].max().item()+1
    num_items = blog_tet_edge_data[0][1].max().item()+1
    embedding_dim = 32
    
    user_indices = blog_tet_edge_data[1][0]
    item_indices = blog_tet_edge_data[1][1]
    
    mf_user_embeddings = nn.Embedding(num_users, embedding_dim)
    mf_item_embeddings = nn.Embedding(num_items, embedding_dim)
    
    nn.init.normal_(mf_user_embeddings.weight, std=0.1)
    nn.init.normal_(mf_item_embeddings.weight, std=0.1)
    
    print(mf_user_embeddings)
    print(mf_item_embeddings)
    
    print("user_indices size is ",user_indices.size()) #-- 38840 -> 유저-아이템 상호작용 횟수 
    print("item_indices size is ",item_indices.size()) #-- 38840 -> 유저-아이템 상호작용 횟수 
    
    #-- 처음 eu 임베딩값은 (610, 32)
    #-- 처음 ei 임베딩값은 (9742, 32)
    mf_user_embedding = mf_user_embeddings(user_indices)
    mf_item_embedding = mf_item_embeddings(item_indices)
    #-- 유저-아이템 상호작용 기반으로 유저 임베딩 아이템 임베딩
    
    print("user_embedding size is {}".format(mf_user_embedding.size())) #-- [38840, 32]
    print("item_embedding size is {}".format(mf_item_embedding.size())) #-- [38840, 32]
    
    predictions = (mf_user_embedding * mf_item_embedding).sum(1) #-- 38840 1차원으로
    print("predictions size is ",predictions.size())
    ```
    

MF 모델의 Foward 과정은 내적연산을 통해 1차원의 임베딩벡터를 만들어 낸다.

→ 이때 유저-아이템 연결관계를 입력으로 받는다.

Light GCN의 Foward  과정은 K번째 홉만큼 메시지 전파 단계를 거친 후에 나온 최종 레이어를 통해 결과를 예측한다.

### GCN 모델 상세

```python
num_users = blog_tet_edge_data[0][0].max().item()+1
num_items = blog_tet_edge_data[0][1].max().item()+1
edge_index = blog_tet_edge_data[0] #-- edge_index 정보가 들어있음

embedding_dim = 32

user_indices = blog_tet_edge_data[1][0]
item_indices = blog_tet_edge_data[1][1]

K = 3
add_self_loops = False

gcn_user_embeddings = nn.Embedding(num_users, embedding_dim)
gcn_item_embeddings = nn.Embedding(num_items, embedding_dim)

nn.init.normal_(gcn_user_embeddings.weight, std=0.1)
nn.init.normal_(gcn_item_embeddings.weight, std=0.1)

# print(gcn_user_embeddings) #-- Embedding(610, 32)
# print(gcn_item_embeddings) #-- Embedding(9742, 32)

edge_index_norm = gcn_norm(
              edge_index, add_self_loops=add_self_loops)

# print(edge_index) #-- 전체 인덱스 정보
# print(edge_index_norm[1]) #-- [0]번 인덱스 == edge_index [1]번 인덱스 == 정규화된 텐서값

emb_0 = torch.cat([gcn_user_embeddings.weight, gcn_item_embeddings.weight]) # E^0
embs = [emb_0]
emb_k = emb_0
```

```python
def forward(self, edge_index: SparseTensor, user_indices, item_indices):
    #-- MF 포워드 과정 --# 
    mf_user_emb = self.mf_user_embeddings(user_indices) # shape: [batch_size, embedding_dim]
    mf_item_emb = self.mf_item_embeddings(item_indices) # shape: [batch_size, embedding_dim]
    mf_prediction = (mf_user_emb * mf_item_emb).sum(dim=1) # element-wise multiplication and sum
    #-- MF 포워드 과정 --#
    
    #-- Light GCN 포워드 과정 --#
    # ... (여기에 이미 있는 코드를 그대로 둡니다.)
    #-- Light GCN 포워드 과정 --#

    # 아래 코드는 MF 예측값과 Light GCN 임베딩을 반환합니다.
    return mf_prediction, gcn_users_emb_final, self.gcn_users_emb.weight, gnc_items_emb_final, self.gnc_items_emb.weight
```

```python
class MFLightGCN(MessagePassing):
    def __init__(self, num_users, num_items, embedding_dim=64, K=3, add_self_loops=False):
        # ... (기존 코드 유지)
          
    def forward(self, user_indices, item_indices, edge_index: SparseTensor):
        #-- MF 포워드 과정 --# 
        mf_user_emb = self.mf_user_embeddings(user_indices)
        mf_item_emb = self.mf_item_embeddings(item_indices)
        
        # 내적을 통한 예측 평점 계산
        mf_prediction = (mf_user_emb * mf_item_emb).sum(dim=1)
        #-- MF 포워드 과정 --#
        
        #-- Light GCN 포워드 과정 --#
        # ... (기존 코드 유지)
        #-- Light GCN 포워드 과정 --#
        
        # 예측 평점도 결과로 반환 (필요에 따라)
        return gcn_users_emb_final, self.gcn_users_emb.weight, gnc_items_emb_final, self.gnc_items_emb.weight, mf_prediction
```