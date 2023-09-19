### **MLP (Multi-Layer Perceptron)**: 기본적인 다층 퍼셉트론을 구현한다. 이는 노드, 엣지, 전역 특성을 업데이트하는 데 사용된다.

---

```python

class MLP(nn.Module):
    def __init__(self, n_in, n_out, hidden=100, nlayers=2, layer_norm=False):
        super().__init__()
        layers = [nn.Linear(n_in, hidden), nn.ReLU()]
        for i in range(nlayers):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden, n_out))
        if layer_norm:
            layers.append(nn.LayerNorm(n_out))
        self.mlp = nn.Sequential(*layers)
				#-- nn.Sequential을 사용하여 신경망을 구성.

    def forward(self, x):
        return self.mlp(x)
```

---

### **EdgeModel**: 엣지 특성을 업데이트하는 모델이다 . 소스 노드, 목표 노드, 엣지 속성, 그리고 전역 속성을 입력으로 받아 엣지 속성을 업데이트한다.

---

```python
# EdgeModel 클래스를 정의한다. 이 클래스는 엣지의 특성을 업데이트하는 역할을 한다.
class EdgeModel(torch.nn.Module):
    # 생성자 함수
    def __init__(self, hidden):
        super(EdgeModel, self).__init__()
        
        # MLP 모델을 초기화한다. 입력 차원은 hidden * 4이고, 출력 차원은 hidden이다.
        # Layer Normalization도 적용합니다.
        self.mlp = MLP(hidden * 4, hidden, layer_norm=True)

    # 순전파 함수
    def forward(self, src, dest, edge_attr, u, batch):
        # src, dest: 엣지의 시작 노드와 끝 노드의 특성. [E, F_x] 형태, E는 엣지의 수.
        # edge_attr: 엣지의 현재 특성. [E, F_e] 형태.
        # u: 그래프의 전역 특성. [B, F_u] 형태, B는 배치 내의 그래프 수.
        # batch: 각 엣지가 어떤 그래프에 속하는지 나타내는 배열. [E] 형태.
        
        # 현재 상태를 계산합니다. src, dest, edge_attr, u를 연결(concatenate)합니다.
        cur_state = torch.cat([src, dest, edge_attr, u[batch]], 1)
        
        # MLP를 통과시켜 엣지 특성을 업데이트하고 반환합니다.
        return edge_attr + self.mlp(cur_state)
```

- **Edge Model 이해를 돕기 위한 예시**
    
    <aside>
    💡 **예시**
    
    - **`src`**: 엣지의 시작점 노드의 정보
    - **`dest`**: 엣지의 끝점 노드의 정보
    - **`edge_attr`**: 엣지 자체의 정보
    - **`u`**: 전체 그래프에 대한 정보
    - **`batch`**: 각 엣지가 어떤 그래프에 속하는지 알려주는 정보
    
    ### **간단한 예시**
    
    예를 들어, 하나의 그래프가 있고 그 그래프에는 2개의 엣지가 있다고 가정
    
    1. 엣지 1: 시작 노드 A, 끝 노드 B
    2. 엣지 2: 시작 노드 C, 끝 노드 D
    
    각 노드와 엣지에는 다음과 같은 정보가 있습니다:
    
    - 노드 A의 정보: **`[1, 2]`**
    - 노드 B의 정보: **`[2, 3]`**
    - 노드 C의 정보: **`[3, 4]`**
    - 노드 D의 정보: **`[4, 5]`**
    - 엣지 1의 정보: **`[1, 1]`**
    - 엣지 2의 정보: **`[2, 2]`**
    
    u : 전체 그래프에 대한 정보는 **`[10, 10]`**이라고 가정
    
    이 때, **`EdgeModel`**의 **`forward`** 함수는 다음과 같이 작동한다:
    
    1. **정보 합치기**: 각 엣지에 대해 시작 노드, 끝 노드, 엣지 정보, 그리고 전체 그래프 정보를 하나로 합친다.
        - 엣지 1: **`[1, 2] (A의 정보) + [2, 3] (B의 정보) + [1, 1] (엣지 1의 정보) + [10, 10] (전체 그래프 정보) = [1, 2, 2, 3, 1, 1, 10, 10]`**
        - 엣지 2: **`[3, 4] (C의 정보) + [4, 5] (D의 정보) + [2, 2] (엣지 2의 정보) + [10, 10] (전체 그래프 정보) = [3, 4, 4, 5, 2, 2, 10, 10]`**
    2. **MLP 모델 통과**: 이 합친 정보를 MLP 모델에 넣어 새로운 엣지 정보를 얻는다. 가정을 위해, 새로운 엣지 정보가 **`[0.5, 0.5]`** (엣지 1)와 **`[1, 1]`** (엣지 2)라고 하겠습니다.
    3. **엣지 정보 업데이트**: 원래의 엣지 정보에 새로운 엣지 정보를 더한다.
        - 엣지 1: **`[1, 1] + [0.5, 0.5] = [1.5, 1.5]`**
        - 엣지 2: **`[2, 2] + [1, 1] = [3, 3]`**
    
    이렇게 업데이트된 엣지 정보가 **`forward`** 함수의 결과로 나온다. 이 정보는 그래프를 더 잘 이해하기 위해 사용된다.
    
    </aside>
    
    ![IMG_86FB74C47890-1.jpeg](https://prod-files-secure.s3.us-west-2.amazonaws.com/60b7a12c-d798-4819-a350-0b02a52409b9/cbe89e2e-5245-49bd-9319-254286e70337/IMG_86FB74C47890-1.jpeg)
    

---

### **NodeModel**: 노드 특성을 업데이트하는 모델이다. 노드 속성, 엣지 인덱스, 엣지 속성, 그리고 전역 속성을 입력으로 받아 노드 속성을 업데이트한다.

---

```python
class NodeModel(torch.nn.Module):
    def __init__(self, hidden):
        super(NodeModel, self).__init__()
        # 첫 번째 MLP 모델 초기화
        self.node_mlp_1 = MLP(hidden * 2, hidden, layer_norm=True)
        # 두 번째 MLP 모델 초기화
        self.node_mlp_2 = MLP(hidden * 3, hidden, layer_norm=True)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: 노드 특성, edge_index: 엣지 인덱스, edge_attr: 엣지 특성
        # u: 전역 특성, batch: 노드가 속한 그래프 배치 정보

        # 엣지 인덱스에서 시작 노드(row)와 목적 노드(col) 분리
        row, col = edge_index

        # 시작 노드의 특성과 엣지 특성을 합침
        out = torch.cat([x[row], edge_attr], dim=1)

        # 첫 번째 MLP 모델을 통과시켜 노드 정보 업데이트
        out = self.node_mlp_1(out)

        # 같은 노드에 연결된 엣지 정보를 집계 (여기서는 합산 사용)
        out = node_aggregation(out, col, dim=0, dim_size=x.size(0))

        # 원래 노드 특성, 집계된 노드 특성, 전역 특성을 합침
        out = torch.cat([x, out, u[batch]], dim=1)

        # 두 번째 MLP 모델을 통과시켜 최종 노드 정보를 얻고 원래 노드 정보에 더함
        return x + self.node_mlp_2(out)
```

- **Node Model 이해를 돕기 위한 예시**
    
    <aside>
    💡 **Node Model 예시**
    
    - `x`: 노드의 특성을 담은 텐서
    - `edge_index`: 엣지의 시작 노드와 끝 노드를 나타내는 인덱스
    - `edge_attr`: 엣지의 특성을 담은 텐서
    - `u`: 그래프 전체에 대한 전역 특성
    - `batch`: 각 노드가 어떤 그래프에 속하는지 나타내는 배치 정보
    
    ### 예시
    
    **이 그래프에는 3개의 노드(A, B, C)와 3개의 엣지(AB, BC, CA)가 있다고 가정**
    
    1. **노드 정보(`x`)**: 각 노드(A, B, C)에는 정보가 있다. 예를 들어, A는 [1, 2], B는 [3, 4], C는 [5, 6]이라는 정보를 가지고 있다고 가정
    2. **엣지 정보(`edge_attr`)**: 엣지(AB, BC, CA)도 정보를 가진다. 예를 들어, AB는 [7], BC는 [8], CA는 [9]라는 정보를 가지고 있다고 가정.
    3. **전체 그래프 정보(`u`)**: 전체 그래프에도 정보가 있을 수 있다. 예를 들어, [10, 11]이라고 가정.
    4. **배치 정보(`batch`)**: 이 예제에서는 하나의 그래프만 있으므로 배치 정보는 필요 없다.
    
    `forward` 함수 실행
    
    1. **노드와 엣지 정보 합치기**: 노드 A와 연결된 엣지 AB의 정보 [1, 2] + [7] = [1, 2, 7]이 된다.
    2. **노드 정보 업데이트 1**: 이 합친 정보를 `node_mlp_1`에 통과시켜 노드 A의 정보를 업데이트한다. 예를 들어, [1, 2, 7]이 [1.5, 2.5]로 업데이트 될 수 있다.
    3. **노드 정보 집계**: 노드 A, B, C 각각에 대해 이런 작업을 하고, 같은 노드에 연결된 엣지 정보를 합친다. 예를 들어, A는 AB와 CA에 연결되어 있으므로, 이 두 엣지의 업데이트된 정보를 합친다.
    4. **노드 정보 업데이트 2**: 마지막으로, 이 합친 정보와 원래 노드 정보, 그리고 전체 그래프 정보를 합쳐서 최종 노드 정보를 업데이트한다.
    - 예를 들어, [1, 2] + [1.5, 2.5] + [10, 11] = [12.5, 15.5]가 될 수 있다.
    
    이렇게 각 노드의 정보가 업데이트되고, 이 정보는 다음 단계에서 그래프의 다른 특성을 예측하거나 분석하는 데 사용된다.
    
    </aside>
    

### **GlobalModel**: 전역 특성을 업데이트하는 모델이다. 노드 속성, 엣지 인덱스, 엣지 속성을 입력으로 받아 전역 속성을 업데이트한다.

```python
class GlobalModel(torch.nn.Module):
    def __init__(self, hidden):
        super(GlobalModel, self).__init__()
        # MLP를 사용하여 전역 특성을 업데이트하는 레이어를 정의한다.
        self.global_mlp = MLP(hidden * 2, hidden, layer_norm=True)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: 각 노드의 특성 [N, F_x], N은 노드 수, F_x는 노드 특성의 차원
        # edge_index: 엣지 인덱스 [2, E], E는 엣지 수
        # edge_attr: 각 엣지의 특성 [E, F_e], F_e는 엣지 특성의 차원
        # u: 전역 특성 [B, F_u], B는 배치(그래프) 수, F_u는 전역 특성의 차원
        # batch: 노드가 어떤 그래프에 속하는지 나타내는 배열 [N]

        # 전역 특성(u)와 노드 특성의 집계(global_aggregation(x, batch, dim=0))를 합친다.
        out = torch.cat([u, global_aggregation(x, batch, dim=0)], dim=1)
        
        # 이 합친 특성을 global_mlp를 통과시켜 전역 특성을 업데이트한다.
        return u + self.global_mlp(out)
```

- **Global Model 이해를 돕기 위한 예시**
    
    <aside>
    💡 두 개의 그래프가 있다고 가정
    첫 번째 그래프는 3개의 노드로 구성
    두 번째 그래프는 2개의 노드로 구성
    
    - 첫 번째 그래프의 노드 특성(`x`):  [[1, 2], [3, 4], [5, 6]]
    - 두 번째 그래프의 노드 특성(`x`): [[7, 8], [9, 10]]
    
    전역 특성(`u`)은 각 그래프에 하나씩 있으므로:
    
    - 첫 번째 그래프의 전역 특성: [[11, 12]]
    - 두 번째 그래프의 전역 특성: [[13, 14]]
    
    이제 이 정보를 `forward` 메서드에 전달하면 다음과 같이 작동한다.
    
    1. 먼저, 각 그래프의 노드 특성을 집계한다. 이 예에서는 `scatter_sum`을 사용하므로:
        - 첫 번째 그래프: (1+3+5 = 9), (2+4+6 = 12)
        - 두 번째 그래프: (7+9 = 16), (8+10 = 18)
    2. 이 집계된 특성과 기존의 전역 특성을 합친다.
        - 첫 번째 그래프: [[11, 12], [9, 12]] -> 합친 후: [[11, 12, 9, 12]]
        - 두 번째 그래프: [[13, 14], [16, 18]] -> 합친 후: [[13, 14, 16, 18]]
    3. 이 합친 특성을 `global_mlp`를 통과시켜 새로운 전역 특성을 얻는다. 이 값은 예를 들어 [[15, 16]]과 [[19, 20]]이 될 수 있다.
    4. 최종적으로, 이 새로운 전역 특성이 출력된다.
    
    이렇게 `GlobalModel`은 각 그래프의 전역 특성을 업데이트하는 역할을 한다.
    
    </aside>
    

### **GNN**: 주된 그래프 신경망 모델이다 . 여러 블록(**`MetaLayer`**)을 통해 노드, 엣지, 그리고 전역 특성을 업데이트한다.

```python

class GNN(torch.nn.Module):
    # 생성자 함수
    def __init__(self, hidden, n_in=1, n_edge=3, n_out=1, decode_on="node", blocks=5):
        super(self.__class__, self).__init__()
        
        # 노드와 엣지의 특성을 인코딩하는 MLP 모델
        self.node_enc = MLP(n_in, hidden, layer_norm=True)
        self.edge_enc = MLP(n_edge, hidden, layer_norm=True)
        
        # 디코더 MLP
        self.decoder = MLP(hidden, n_out)
        
        # MetaLayer를 사용한 연산 블록
        self.ops = nn.ModuleList(
            [
                MetaLayer(EdgeModel(hidden), NodeModel(hidden), GlobalModel(hidden))
                for _ in range(blocks)
            ]
        )
        
        # 디코딩 대상 설정 ("node" 또는 "global")
        self.decode_on = decode_on
        self.hidden = hidden

    # 순전파 함수
    def forward(self, graph):
        # 노드 특성 중 M14만 선택하여 인코딩
        x = self.node_enc(graph.x[:, [3]])
        
        # 노드 간의 상대적 위치
        pos = graph.x[:, :3]
        
        # 인접 행렬
        adj = graph.edge_index
        
        # 엣지 특성 인코딩
        e = self.edge_enc(pos[adj[0]] - pos[adj[1]])

        # 전역 특성 초기화
        u = torch.zeros(
            graph.batch[-1] + 1, self.hidden, device=x.device, dtype=torch.float32
        )
        
        # 배치 정보
        batch = graph.batch

        # 각 연산 블록을 통과
        for op in self.ops:
            x, e, u = op(x, adj, e, u, batch)

        # 디코딩
        if self.decode_on == "node":
            out = self.decoder(x)
        elif self.decode_on == "global":
            out = self.decoder(u)

        return out
```

- **GNN Model 예시**
    
    <aside>
    💡 **예시**
    
    1. **노드 특성 인코딩**:
    
    <aside>
    💡 노드의 특성 벡터가 **`[a, b, c, M14, ...]`** 형태라고 하면, 이 중에서 **`M14`**만을 추출하여 노드 인코딩에 사용한다. 이렇게 특정 특성만을 선택하는 이유는 다양할 수 있으며, 문제의 도메인이나 데이터의 특성에 따라 결정된다. M14가 예를 들어 별의 질량이나 다른 중요한 특성을 나타낸다면, 그것만을 선택하여 노드를 인코딩하는 것이 유용할 수 있다
    
    </aside>
    
    - 입력 그래프의 노드 특성 중에서 M14만을 선택하여 `node_enc` MLP 모델을 통과시킨다.
    - 예: 만약 그래프의 노드 특성이 `[1, 2, 3, 4]`, `[5, 6, 7, 8]`이라면, M14는 `4`와 `8`이다. 이들을 `node_enc`에 통과시켜 새로운 노드 특성을 얻는다.
    1. **엣지 특성 인코딩**:
        - 노드 간의 상대적 위치를 계산하고, 이를 `edge_enc` MLP 모델을 통과시켜 엣지 특성을 얻는다.
        - 예: 노드 A와 B의 위치가 각각 `[1, 2, 3]`, `[4, 5, 6]`이라면, 상대적 위치는 `[3, 3, 3]`이다.
    2. **전역 특성 초기화**:
        - 전역 특성 `u`를 0으로 초기화한다.
    3. **연산 블록 통과**:
        - 노드, 엣지, 전역 특성을 연산 블록 (`MetaLayer`)에 통과시켜 업데이트한다.
        - 예: 첫 번째 연산 블록을 통과한 후, 노드 특성은 `[new_x1, new_x2, ...]`, 엣지 특성은 `[new_e1, new_e2, ...]`, 전역 특성은 `new_u`로 업데이트된다.
    4. **디코딩**:
        - `decode_on` 변수에 따라 노드 레벨 또는 전역 레벨에서 디코딩을 수행한다.
        - 예: `decode_on = "node"`이면, 업데이트된 노드 특성을 `decoder`에 통과시켜 최종 출력을 얻는다.
    
    이러한 과정을 통해 그래프의 노드, 엣지, 전역 특성을 학습하고, 디코딩을 통해 원하는 출력을 얻는다. 이는 그래프 데이터에 내재된 복잡한 패턴을 파악하는 데 유용하다.
    
    </aside>
    

### **GNNAllocation**: 노드와 전역 특성을 업데이트하는 데 사용되는 두 개의 GNN 모델(**`allocator`**와 **`predictor`**)을 포함한다. 이 모델은 노드의 속성을 업데이트한 후 전역 속성을 예측한다.

```python
class GNNAllocation(nn.Module):
    """GNN of the form:
    z_i = f_{in}(x_i)
    For k in Range(n_messages):
        z_i = z_i + g_k(z_i, sum_{j->i} h_k(z_i, z_j))

    y_i = f_{out}(z_i)
    """

    def __init__(
        self,
        n_in,  # e.g., position, Mass
        n_out,  # e.g., Om, s8, etc
        n_v=100,
        n_e=100,
        dim=3,
        hidden=100,
        nlayers=2,
        use_edge_model=False,
        n_messages=5,
        layer_norm=False,
    ):
        super(self.__class__, self).__init__()
        self.allocator = GNN(
            hidden=hidden, n_out=1, decode_on="node", blocks=n_messages
        )
        self.predictor = GNN(
            hidden=hidden, n_out=n_out, decode_on="global", blocks=n_messages
        )

    def forward(self, graph, snr_model):
				#-- 원본 그래프 복제
        orig_graph = graph.clone()
        n = graph.x.shape[0]

				#-- 노드 특성 추출
        M14 = graph.x[:, [3]].clone()
        true_M = torch.log10(M14 * 1e14)

        true_z = graph.x[:, [4]].clone()
        time1 = torch.ones_like(true_M)
				
				#-- 관측 표준편차 계산
				#-- snr_model: 신호 대 잡음비(SNR) 모델로, 노드의 속성을 업데이트하는 데 사용
        obs_std1 = snr_model(torch.cat((true_M, time1, true_z), dim=1))
        Mstd1 = torch.exp(np.log(10) * obs_std1[:, [0]])
        zstd1 = torch.exp(np.log(10) * obs_std1[:, [1]])

        graph = orig_graph.clone()
				#-- 노드 특성을 노이즈가 추가된 값으로 업데이트
        graph.x[:, [3]] += torch.randn_like(Mstd1) * Mstd1
        graph.x[:, [4]] += torch.randn_like(zstd1) * zstd1

				#-- 각 노드에 할당된 시간 계산
        time2 = (
            time1 + torch.sigmoid(self.allocator(graph) - 3) * 59
        )  # Up to a maximum of 60 minutes per source.

				#-- 새로운 관측 표준편차 계산
				#-- snr_model: 신호 대 잡음비(SNR) 모델로, 노드의 속성을 업데이트하는 데 사용
        obs_std2 = snr_model(torch.cat((true_M, time2, true_z), dim=1))
        Mstd2 = torch.exp(obs_std2[:, [0]])
        zstd2 = torch.exp(obs_std2[:, [1]])

        graph = orig_graph
        graph.x = torch.cat(
            (
                graph.x[:, :3],
                graph.x[:, [3]] + torch.randn_like(Mstd2) * Mstd2,
                graph.x[:, [4]] + torch.randn_like(zstd2) * zstd2,
            ),
            dim=1,
        )
				#-- 그래프의 전역 특성 예측
        predictions = self.predictor(graph)
        return predictions, {
            "time": time2,
            "Mstd1": Mstd1,
            "zstd1": zstd1,
            "Mstd2": Mstd2,
            "zstd2": zstd2,
        }
```

- **이해를 돕기 위한 예시**
    
    <aside>
    💡 `GNNAllocation` 클래스의 `forward` 함수를 중심으로 예시
    
    ### 예시
    
    - 가정: 그래프에는 3개의 노드가 있고, 각 노드의 `M14` 값은 `[1.0, 2.0, 3.0]`, `true_z` 값은 `[0.1, 0.2, 0.3]`이라고 가정
    - `snr_model`은 단순히 입력에 0.1을 더하는 함수라고 가정합니다.
    
    ### 1단계: 원본 그래프 복제
    
    - `orig_graph`에 원본 그래프를 저장합니다.
    
    ### 2단계: 노드 특성 추출
    
    - `M14 = [1.0, 2.0, 3.0]`
    - `true_z = [0.1, 0.2, 0.3]`
    
    ### 3단계: 관측 표준편차 계산
    
    - `obs_std1 = snr_model([1.0, 0.1], [2.0, 0.2], [3.0, 0.3]) = [1.1, 0.2, 3.1, 0.3]`
    
    ### 4단계: 노드 특성을 노이즈가 추가된 값으로 업데이트
    
    - 랜덤 노이즈를 추가하여 `M14`와 `true_z`를 업데이트합니다.
        - 예: `M14_new = [1.05, 2.1, 3.2]`
        - 예: `true_z_new = [0.11, 0.21, 0.33]`
    
    ### 5단계: 각 노드에 할당된 시간 계산
    
    - `allocator` 모델을 사용하여 각 노드에 할당된 시간을 계산합니다.
        - 예: `time2 = [10, 20, 30]` (분)
    
    ### 6단계: 새로운 관측 표준편차 계산
    
    - `obs_std2 = snr_model([1.05, 10, 0.11], [2.1, 20, 0.21], [3.2, 30, 0.33])`
        - 예: `obs_std2 = [1.15, 10.1, 0.12, 2.2, 20.1, 0.22, 3.3, 30.1, 0.34]`
    
    ### 7단계: 그래프의 전역 특성 예측
    
    - `predictor` 모델을 사용하여 그래프의 전역 특성을 예측합니다.
        - 예: `predictions = [0.5, 0.6]`
    
    이렇게 `GNNAllocation` 클래스는 각 노드의 특성을 업데이트하고, 각 노드에 할당된 시간을 계산한 뒤, 그래프의 전역 특성을 예측하는 역할을 합니다.
    
    </aside>
    

### 주요 클래스와 메서드

1. **MLP (Multi-Layer Perceptron)**: 기본적인 피드포워드 신경망을 구현한다. 다양한 계층과 활성화 함수를 포함할 수 있다.
2. **EdgeModel**: 엣지 특성을 업데이트하는 모델이다. 노드와 엣지, 그리고 전역 특성을 입력으로 받아 엣지 특성을 업데이트한다.
3. **NodeModel**: 노드 특성을 업데이트하는 모델이다. 노드와 엣지, 그리고 전역 특성을 입력으로 받아 노드 특성을 업데이트한다.
4. **GlobalModel**: 전역 특성을 업데이트하는 모델이다. 노드와 엣지, 그리고 전역 특성을 입력으로 받아 전역 특성을 업데이트한다.
5. **GNN**: 위의 모델들을 조합하여 전체 그래프 신경망을 구성한다. 노드, 엣지, 전역 특성을 업데이트하는 여러 계층을 포함한다.
6. **GNNAllocation**: `GNN` 모델을 사용하여 특정 작업을 수행한다. 예를 들어, 노드의 특성(`M14`, `true_z` 등)을 기반으로 예측을 수행하거나, 노드에 할당된 시간(`time`)을 업데이트한다.

### 얻을 수 있는 것

1. **특성 업데이트**: 노드, 엣지, 전역 특성이 업데이트되며, 이는 그래프 구조 내에서 정보를 더 잘 표현할 수 있게 한다.
2. **예측**: `GNNAllocation` 클래스를 통해, 그래프의 특성을 기반으로 다양한 예측을 수행할 수 있다. 예를 들어, 노드에 할당된 시간을 업데이트하거나, 다른 목표 변수를 예측할 수 있다.
3. **효율성**: 코드는 효율적인 그래프 연산을 사용하여 대규모 그래프에 대한 빠른 계산을 가능하게 한다.
4. **모듈성**: 각 모델(`EdgeModel`, `NodeModel`, `GlobalModel`)은 독립적으로 구현되어 있어, 다른 문제나 설정에 쉽게 적용할 수 있.

이러한 과정과 모델을 통해, 복잡한 그래프 구조에서 유용한 특성을 추출하고, 그를 기반으로 다양한 예측이나 분석을 수행할 수 있다.