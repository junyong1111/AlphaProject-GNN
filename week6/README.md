### **심층 강화 학습과 그래프 신경망의 만남: 라우팅 최적화 사용 사례 살펴보기**

- **Deep Reinforcement Learning meets Graph Neural Networks: exploring a routing optimization use case**
    
    [1910.07421.pdf](https://prod-files-secure.s3.us-west-2.amazonaws.com/60b7a12c-d798-4819-a350-0b02a52409b9/3f1b451b-b906-4b21-b075-233b8d3576f5/1910.07421.pdf)

## ***Abstract***

**Abstract:**
Deep Reinforcement Learning (DRL)은 결정 및 자동 제어 문제에서 중요한 개선을 보였다. 따라서 DRL은 자율 주행 네트워크에서 많은 중요한 최적화 문제 (예: 라우팅)를 효과적으로 해결하는 유망한 기술이다. 그러나 기존의 DRL 기반 솔루션은 표준 신경망 (예: 완전 연결, 합성곱)을 사용하는 **네트워킹에 일반화**하지 못하며, 이는 그래프로 구조화된 정보에서 학습하기에 적합하지 않다.

이 논문에서는 일반화를 가능하게 하기 위해 Graph Neural Networks (GNN)를 DRL 에이전트에 통합하고 문제 특정 작업 공간을 설계했다. GNN은 다양한 크기와 구조의 그래프에 대해 일반화하기 위해 본질적으로 설계된 Deep Learning 모델이다. 이를 통해 제안된 GNN 기반 DRL 에이전트는 임의의 네트워크 토폴로지에서 학습하고 일반화할 수 있다. 우리는 광 네트워크에서의 라우팅 최적화 사용 사례에서 **DRL+GNN 에이전트를 테스트하고 180개와 232개의 보이지 않는 합성 및 실제 네트워크 토폴로지에서 평가**했다. 결과는 DRL+GNN 에이전트가 훈련 중에 볼 수 없었던 토폴로지에서 최첨단 솔루션을 능가할 수 있음을 보였다.

**요약:**
이 논문은 DRL과 GNN을 결합하여 네트워크 최적화 문제, 특히 광 네트워크에서의 라우팅 최적화를 탐구했다. 제안된 **DRL+GNN** 아키텍처는 학습 중에 본 적 없는 임의의 네트워크 토폴로지에서 일반화하고 성능을 발휘할 수 있음을 실험적으로 확인했다.

## I. INTRODUCTION

 최근 몇 년 동안 인더스트리 4.0 및 IoT와 같은 산업의 발전과 사회적 행동의 변화로 인해 차량 네트워크, AR/VR, 실시간 커뮤니케이션과 같은 최신 네트워크 애플리케이션이 등장했다. 이러한 애플리케이션은 백본 네트워크에 새로운 요구 사항을 부과하며, 네트워크 사업자는 네트워크 리소스를 효율적으로 관리하여 높은 처리량과 짧은 지연 시간을 보장하고 고객의 서비스 품질 및 서비스 수준 협약을 충족해야 한다. 일반적으로 이는 전문 지식이나 정수 선형 프로그래밍(ILP) 또는 제약 조건 프로그래밍과 같은 솔버를 사용하여 달성된다.

 이 연구에서 저자는 **그래프 신경망(GNN)과 심층 강화 학습(DRL) 에이전트를 결합**하여 네트워크 최적화 문제를 해결하며, 특히 광 네트워크의 라우팅 최적화에 중점을 둔다. DRL 에이전트에 통합된 GNN은 메시지 전달 신경망(MPNN)에서 영감을 받았으며 다양한 네트워크 토폴로지에서 네트워크 링크와 트래픽 흐름 간의 관계에 대한 관련 정보를 캡처하도록 설계되었다.

 에이전트의 평가 결과, 일반화 기능 측면에서 **최첨단 심층 강화 학습(DRL) 알고리즘보다 뛰어난 성능**을 발휘하는 것으로 나타났다. 또한 232개의 서로 다른 실제 네트워크 토폴로지에서 제안한 **DRL+GNN 아키텍처를 테스트한 결과, 학습 중에 볼 수 없었던 네트워크에서도 뛰어난 성능을 달성**했다. 마지막으로, 아키텍처 일반화의 한계를 조사하고 확장성 속성에 대해 논의한다.

 제안된 DRL+GNN 아키텍처에서 언급된 기능의 조합은 휴리스틱이나 선형 최적화에 의존하는 기존 접근 방식에 비해 심층 강화 학습(DRL)에 기반하고 비용 효율적인 혁신적인 네트워킹 솔루션의 개발을 촉진할 수 있는 잠재력을 가지고 있다. 또한 실험에 사용된 DRL+GNN 에이전트의 토폴로지, 스크립트 및 소스 코드는 공개적으로 액세스할 수 있어 연구 커뮤니티의 투명성과 재현성을 촉진한다

<aside>
💡 최근 인더스트리 4.0 및 IoT의 발전과 사회적 행동의 변화로 인해 새로운 네트워크 애플리케이션이 등장하였고, 이로 인해 백본 네트워크에 높은 처리량과 짧은 지연 시간과 같은 새로운 요구 사항이 생겼다. 네트워크 사업자들은 이러한 요구 사항을 충족하기 위해 다양한 방법을 사용하고 있다. 이 연구에서는 그래프 신경망(GNN)과 심층 강화 학습(DRL)을 결합하여 네트워크 최적화 문제, 특히 광 네트워크의 라우팅 최적화에 접근한다. 제안된 DRL+GNN 아키텍처는 다양한 네트워크 토폴로지에서 뛰어난 성능을 보여주며, 기존의 휴리스틱이나 선형 최적화 방식보다 더 효과적이다. 이 연구의 결과물은 공개적으로 액세스 가능하여 연구의 투명성과 재현성을 증진시킨다.

</aside>

## II. BACKGROUND

 저자들은 컴퓨터 네트워크 시나리오를 모델링하고 네트워크 운영을 최적화하기 위해 그래프 신경망(GNN)과 심층 강화 학습(DRL)을 결합하는 솔루션을 제안한다. GNN은 그래프 구조의 데이터를 처리하도록 설계된 신경망 아키텍처이며, DRL은 에이전트가 계산 집약적인 알고리즘 없이도 과거의 최적화를 기반으로 네트워크를 효율적으로 운영할 수 있게 해준다. 저자들은 GNN이 초기 상태를 그래프의 요소에 연결하고 그래프의 연결에 따라 반복적으로 상태를 업데이트하여 네트워크 시나리오를 효과적으로 모델링할 수 있다고 설명한다.

***A. Graph Neural Networks***

- GNNs는 그래프로 구조화된 데이터에서 작동한다. GNN에서 메시지 전달 단계 동안 각 노드는 이웃 노드로부터 메시지를 받는다. 이 메시지들은 그래프 내의 노드 쌍의 숨겨진 상태에 메시지 함수를 적용함으로써 생성된다. 그 후, 이들은 집계 함수를 사용하여 결합되며, 업데이트 함수는 각 노드에 대한 새로운 숨겨진 상태를 계산한다.
- GNNs는 데이터가 그래프로 구조화된 다양한 도메인에서 중요한 성능을 보여왔다. 컴퓨터 네트워크가 기본적으로 그래프로 표현되기 때문에, GNNs는 전통적인 신경망 아키텍처에 비해 네트워크 모델링에 독특한 이점을 제공한다.

***B. Deep Reinforcement Learning***

- DRL 알고리즘은 최적화 문제에서 목적 함수를 최대화하기 위한 장기 전략을 학습하는 것을 목표로 한다. DRL 에이전트는 사전 지식 없이 시작하고 상태와 행동 공간을 반복적으로 탐색함으로써 최적의 전략을 학습한다. 목표는 에피소드의 끝까지 누적 보상을 최대화하는 전략을 찾는 것이다.
- Q-러닝은 에이전트가 정책을 학습하게 하는 특정 RL 알고리즘이다. 이 알고리즘은 누적 보상이 가장 높은 상태와 행동을 기반으로 전략을 생성한다.

![스크린샷 2023-10-08 오후 5.26.08.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/60b7a12c-d798-4819-a350-0b02a52409b9/dd5cd08f-4bfe-400e-8587-6b65a0ef46f1/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2023-10-08_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_5.26.08.png)

<aside>
💡 이 방정식은 Q-러닝(Q-learning) 알고리즘에서 사용되는 업데이트 규칙을 나타냅니다. 이 방정식은 q-테이블(q-table)에 있는 Q-값을 업데이트하는 데 사용됩니다. 여기서 q-테이블은 특정 상태에서 특정 동작을 취할 때 예상되는 누적 보상을 나타내는 것입니다.

방정식의 구성 요소를 자세하게 설명해드리겠습니다:

1. 현재 상태와 동작: st, at
    - st: 현재 상태 (state)
    - at: 현재 동작 (action)
2. 학습률과 할인율: α, γ
    - α(알파): 학습률로써, 새로운 정보를 얼마나 반영할지 결정합니다.
    - γ(감마): 할인율로써, 다음 상태 및 동작의 최대 Q-값에 대한 가중치 조절에 사용됩니다.
3. 즉시 보상과 다음 상태 및 동작의 최대 Q-값:
r(st, at): 해당 시간 단계에서 받은 바로 가까운 보상입니다.
max a'Q(s't,a'): 다음 시간 단계(t+1)에서 가능한 모든 동작(a') 중 최대 값을 선택하여 계산된 값으로,
다음 상태(s't)에서 최선의 동작을 선택하는 데 사용됩니다.
4. Q-값 업데이트:
    - Q(st, at): 현재 상태와 동작에 대한 기존의 Q-값
    - α(r(st, at) + γ max a'Q(s't,a')): 즉시 보상과 다음 상태 및 동작의 최대 Q-값을 합친 값으로,
    새로운 정보를 반영하여 업데이트된 값을 나타냅니다.
    - = : 등호는 이전에 계산된 기존의 Q-값과 새로 계산된 값을 결합한다는 의미입니다.

이 방정식은 현재 상태와 동작에 대한 예상되는 누적 보상인 q-value를 갱신하는 과정을 나타내며, 알파(α) 학습률과 감마(γ) 할인율을 통해 어느 정도까지 이전 정보를 유지하고 신규 정보를 반영할 지 조절합니다.

</aside>

<aside>
💡 농장 주인이 자신의 작물을 효율적으로 관리하기 위해 Q-러닝 알고리즘을 사용한다고 가정해보겠습니다.

1. 현재 상태와 동작:
    - st: 농장의 현재 상태 (예: 날씨, 온도, 습도 등)
    - at: 농부가 취할 수 있는 동작 (예: 물주기, 비료 추가 등)
2. 학습률과 할인율:
    - α(알파): 학습률로써, 새로운 정보를 얼마나 반영할지 결정합니다.
    - γ(감마): 할인율로써, 다음 시간 단계에서 예상되는 보상에 대한 가중치 조절에 사용됩니다.
3. 즉시 보상과 다음 상태 및 동작의 최대 Q-값:
r(st, at): 해당 시간 당 작업으로 인해 발생하는 바로 가까운 이익/손실입니다.
max a'Q(s't,a'): 다음 시간 당 가능한 모든 작업(a') 중에서 기대되는 최대 값을 선택하여 계산된 값으로,
그 후에 나타난 상황(s't)에서 최선의 작업을 선택하는 데 사용됩니다.
4. Q-값 업데이트:
    - Q(st, at): 현재 상태와 동작에 대한 기존의 Q-값
    - α(r(st, at) + γ max a'Q(s't,a')): 즉시 보상과 다음 상태 및 동작의 최대 Q-값을 합친 값으로,
    새로운 정보를 반영하여 업데이트된 값을 나타냅니다.
    - = : 등호는 이전에 계산된 기존의 Q-값과 새로 계산된 값을 결합한다는 의미입니다.

이 방정식은 농장 주인이 현재 농장 상황에서 어떤 작업을 선택해야 하는지를 결정하기 위해 사용됩니다. 예를 들어, 날씨가 맑고 습도가 낮으면 "물주기"라는 동작을 선택할 수 있습니다. 그리고 해당 작업으로 인해 발생하는 바로 가까운 이익/손실(r(st, at))와 다음 시간 당 가능한 모든 작업 중에서 예상되는 최대 보상(max a'Q(s't,a'))을 고려하여 현재 상태와 동작에 대한 q-value(Q(st, at))를 갱신합니다. 이런 식으로 알고리즘이 반복되면서 점점 효율적인 관리 방법이 학습되며 자동화된 결정 메커니즘이 구축됩니다.

</aside>

## III. NETWORK OPTIMIZATION SCENARIO

 논문에서 저자는 **광전송 네트워크(OTN)의 라우팅 문제를 해결하기 위해 그래프 신경망(GNN) 기반 심층 강화 학습(DRL) 에이전트를 사용하는 방법**에 대해 조사한다. 제어 평면에 위치한 DRL 에이전트가 들어오는 트래픽 수요에 대한 라우팅 결정을 내리는 소프트웨어 정의 네트워킹 기반 네트워크 시나리오에 중점을 둔다. 이 최적화 문제는 광 네트워킹 분야에서 광범위하게 연구되어 왔으며, 저자들은 이러한 맥락에서 제안된 접근 방식의 잠재력을 탐구하고자 한다.

 OTN(광 전송 네트워크) 시나리오에서 DRL 에이전트는 전기 도메인에서 작동하며 재구성 가능한 광 추가 드롭 멀티플렉서(ROADM)를 나타내는 노드와 이를 연결하는 사전 정의된 광 경로로 구성된 논리적 토폴로지를 기반으로 라우팅 결정을 내린다. 에이전트는 대역폭 요구 사항이 서로 다른 트래픽 수요를 수신하고 각 수요의 소스와 목적지를 연결하는 일련의 광경로인 엔드투엔드 경로를 선택한다. 트래픽 수요는 특정 대역폭 요구 사항을 가진 ODUk(Optical Data Unit)의 요청으로 정의된다.

 주어진 시나리오에서 라우팅 문제는 들어오는 각 소스-대상 트래픽 수요에 대해 최적의 라우팅 정책을 찾는 것을 말한다. 목표는 장기적으로 네트워크에 할당된 트래픽 양을 최대화하는 것이다. 선택한 종단 간 경로를 구성하는 광 경로에 사용 가능한 용량이 충분하면 수요가 적절하게 할당된 것으로 간주된다. 에이전트는 중요한 네트워크 리소스를 식별하고, 향후 트래픽 수요의 불확실성을 처리하고, 순차적 라우팅 결정 및 에피소드 중 트래픽 수요를 분할하거나 경로를 변경할 수 없는 등의 제약 조건을 준수해야 하는 문제에 직면한다.

 OTN 최적화 문제에 대한 최적의 솔루션은 마르코프 결정 과정(MDP)이라는 수학적 프레임워크를 풀면 얻을 수 있다. 동적 프로그래밍과 같은 기법을 사용하여 가능한 모든 네트워크 상태와 그 전이 확률을 고려함으로써 MDP를 반복적으로 풀 수 있다. 그러나 상태 공간의 기하급수적인 증가로 인해 대규모의 복잡한 최적화 문제에서는 **MDP를 최적으로 푸는 것이 비현실적이며, 상당한 계산 리소스와 시간이 필요하다.**

## IV. GNN-BASED DRL AGENT DESIGN

이 섹션에서는 네트워크 최적화 문제를 해결하기 위해 **심층 강화 학습(DRL)과 그래프 신경망(GNN)을 결합한 DRL+GNN 아키텍처에 대해 설명**한다. GNN을 기반으로 하는 DRL 에이전트는 후보 경로에 대한 수요 할당과 같이 네트워크 토폴로지에서 수행해야 할 작업을 결정한다. 네트워크 토폴로지 및 링크 기능을 포함하는 환경은 에이전트의 작업을 평가하고 학습 과정을 안내하기 위해 보상을 생성한다.

 제안된 DRL+GNN 아키텍처의 학습 프로세스에는 에이전트가 네트워크 상태 관찰을 수신하고 GNN을 사용하여 그래프 표현을 구성하는 반복적 접근 방식이 포함된다. 링크 숨겨진 상태는 입력 특징과 평가 중인 라우팅 작업을 기반으로 초기화된다. 그런 다음 링크 숨겨진 상태 간에 반복적인 메시지 전달 알고리즘이 적용되어 새로운 숨겨진 상태가 생성되고, 이 숨겨진 상태는 글로벌 숨겨진 상태로 집계된다. 이 글로벌 숨겨진 상태는 DNN에 의해 처리되고 DNN은 q-값 추정치를 출력합니다. DRL 에이전트는 제한된 액션 세트에서 가장 높은 q-값을 가진 액션을 선택한다.

1. **문제 소개**:
    - 네트워킹을 위한 **전통적인 DRL 기반 솔루션**은 **일반화에 실패**하는 경향이 있다. 즉, 훈련 중에 관찰되지 않은 네트워크 토폴로지에 적용될 때 올바르게 작동하지 않을 수 있다.
    - 이러한 제한은 대부분의 DRL 기반 네트워킹 솔루션에서 그래프로 구조화된 데이터에서 학습하기에 적합하지 않은 표준 신경망(예: 완전 연결 또는 합성곱 신경망)을 사용하기 때문에 발생한다.
2. **GNN을 사용한 솔루션**:
    - 이 논문은 그래프 신경망(GNN)을 DRL 에이전트에 통합하는 것을 제안한다. GNN은 다양한 크기와 구조의 그래프에 대해 일반화하기 위해 본질적으로 설계된 딥러닝 모델이다.
    - GNN을 DRL 에이전트에 통합함으로써 결과적인 에이전트는 훈련 중에 보지 못한 어떠한 네트워크 토폴로지에서도 학습하고 일반화할 수 있다.
3. **평가 및 결과**:
    - DRL+GNN 에이전트는 광 네트워크에서의 라우팅 최적화 시나리오에서 테스트되었다. 훈련 중에 보지 못한 합성 및 실제 네트워크 토폴로지에서 평가되었다.
    - 결과는 DRL+GNN 에이전트가 훈련 중에 볼 수 없었던 토폴로지에서 최첨단 솔루션을 능가하는 것으로 나타났다.
4. **링크 실패에 대한 복원력 사용 사례**:
    - DRL+GNN 에이전트의 네트워크 토폴로지 변화에 대한 적응성을 평가하는 사용 사례에 대해서도 언급하며, 특히 링크 실패 시나리오에서의 적응성에 중점을 둔다.
    - 결과는 최대 10개의 동시 링크 실패가 발생하더라도 DRL+GNN 에이전트가 이론적 기준선보다 더 나은 성능을 유지할 수 있음을 보여준다.
5. **배포에 대한 논의**:
    - DRL이 자율 주행 네트워크의 맥락에서 성공하기 위해서는 일반화 능력이 있어야 한다. 고객의 네트워크에서 DRL 에이전트를 훈련시키는 것은 잠재적인 중단 때문에 실행 가능하지 않을 수 있다. 그러나 일반화 능력이 있으면 DRL 에이전트는 통제된 환경에서 훈련을 받고 본적 없는 네트워크나 시나리오에 배포될 수 있다.
    - 이 논문은 DRL+GNN 아키텍처를 기반으로 한 제품의 훈련 및 배포 과정을 설명하며, 실제 환경에서의 배포를 위한 일반화의 중요성을 강조한다.
    **

## V. EXPERIMENTAL RESULTS

1. **방법론**:
    - DRL+GNN 에이전트의 평가는 두 세트의 실험으로 나뉜다.
    - 첫 번째 세트는 솔루션의 성능과 일반화 능력을 이해하기 위한 것이다. 깊은 분석을 위해 두 가지 특정 네트워크 시나리오가 선택된다.
    - 기준선은 광 전송 네트워크(OTN)에서 라우팅 최적화를 위한 최첨단 솔루션인 참조된 연구에서 제안된 DRL 기반 시스템을 사용하여 설정한다.
    - 나중에 솔루션은 실제 네트워크 토폴로지에서 계산 시간과 일반화 능력 측면에서의 확장성을 분석하기 위해 평가된다.
    - OTN 최적화 문제에 대한 최적의 MDP 솔루션을 찾는 것은 그 복잡성 때문에 어렵다. 대안으로 DRL+GNN 에이전트의 성능은 이론적 유체 모델과 비교된다. 이 모델은 트래픽 요구 사항이 사용 가능한 용량을 기반으로 후보 경로로 분할될 수 있다고 고려한다.
2. **합성 및 실제 토폴로지에서의 실험 결과**:
    - DRL+GNN 아키텍처는 합성 및 실제 토폴로지 모두에서 테스트된다.
    - 결과는 아키텍처가 특정 토폴로지에서 성능 문제를 가지고 있음을 나타낸다. 성능 저하는 트레이닝 중에 사용된 토폴로지의 네트워크 특성과의 차이 때문으로 귀속된다.
    - 링크 특성은 정규화되고, 트래픽 요구 사항은 항상 같은 대역폭 값을 가진다. 그러나 네트워크 토폴로지의 변화는 DRL 에이전트의 성능에 직접적인 영향을 미친다.
    - 각 토폴로지 크기에 대한 다양한 토폴로지 지표가 제시된다. 에지의 중간성은 특정 방식으로 계산되며, Theoretical Fluid 모델과 비교한 DRL+GNN의 성능도 보여진다.
    - 합성 토폴로지는 Nsfnet과 유사한 노드 정도를 가지도록 생성된다. 그러나 토폴로지가 커질수록 다른 지표들이 시작된다.
    - 실제 토폴로지의 경우, 결과는 DRL+GNN 아키텍처의 성능이 Nsfnet와 크게 다른 토폴로지에서 더 나쁘다는 것을 보여준다.
3. **일반화에 대한 논의**:
    - DRL+GNN 에이전트의 성능은 토폴로지의 차이에 영향을 받는다. 토폴로지가 Nsfnet와 얼마나 다른지에 따라 DRL의 성능이 떨어진다.
    - 이러한 토폴로지에 대한 일반화 능력을 향상시키기 위한 여러 개선 방안이 제안된다. 이러한 방안에는 트레이닝 세트에 다른 특성을 가진 토폴로지를 포함시키는 것, 전통적인 딥 러닝 기술을 사용하여 DRL+GNN 아키텍처를 개선하는 것, 그리고 다양한 방법을 사용하여 인접한 링크의 정보를 집계하는 것이 포함된다.

이 섹션은 DRL+GNN 에이전트의 실험 결과에 대한 포괄적인 분석을 제공하며, 그 강점과 개선 영역을 강조한다. 결과는 DRL+GNN 에이전트의 라우팅 최적화에 대한 잠재력을 보여주며, 특히 다양한 네트워크 토폴로지에 적용될 때 그 잠재력을 보여준다.

- CODE
    
    **라이브러리 import**
    
    ```python
    import numpy as np  # 넘파이 라이브러리를 가져옵니다.
    import gym  # OpenAI의 강화학습 환경 라이브러리를 가져옵니다.
    import gc  # 가비지 컬렉터를 위한 라이브러리를 가져옵니다.
    import os  # 운영체제와 상호작용하기 위한 라이브러리를 가져옵니다.
    import sys  # 시스템 파라미터와 함수에 접근하기 위한 라이브러리를 가져옵니다.
    import gym_environments  # 사용자 정의 강화학습 환경을 가져옵니다.
    import random  # 랜덤 함수를 위한 라이브러리를 가져옵니다.
    import mpnn as gnn  # 사용자 정의 그래프 신경망 라이브러리를 가져옵니다.
    import tensorflow as tf  # 텐서플로우 라이브러리를 가져옵니다.
    from collections import deque  # 덱 자료구조를 가져옵니다.
    import multiprocessing  # 멀티프로세싱을 위한 라이브러리를 가져옵니다.
    import time as tt  # 시간 관련 함수를 위한 라이브러리를 가져옵니다.
    import glob  # 파일 경로와 관련된 함수를 위한 라이브러리를 가져옵니다.
    ```
    
    **초기설정**
    
    ```python
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # CUDA를 사용하지 않도록 설정합니다.
    
    ENV_NAME = 'GraphEnv-v1'  # 사용할 환경의 이름을 설정합니다.
    graph_topology = 0  # 그래프 토폴로지를 설정합니다. 여기서는 NSFNET을 의미합니다.
    SEED = 37  # 랜덤 시드 값을 설정합니다.
    ITERATIONS = 10000  # 반복 횟수를 설정합니다.
    TRAINING_EPISODES = 20  # 훈련 에피소드 수를 설정합니다.
    EVALUATION_EPISODES = 40  # 평가 에피소드 수를 설정합니다.
    FIRST_WORK_TRAIN_EPISODE = 60  # 첫 작업 훈련 에피소드를 설정합니다.
    
    MULTI_FACTOR_BATCH = 6  # 훈련에 사용될 배치의 수를 설정합니다.
    TAU = 0.08  # 소프트 가중치 복사에 사용되는 값입니다.
    
    differentiation_str = "sample_DQN_agent"  # DQN 에이전트의 이름을 설정합니다.
    checkpoint_dir = "./models"+differentiation_str  # 체크포인트를 저장할 디렉토리를 설정합니다.
    store_loss = 3  # 손실을 저장할 배치 간격을 설정합니다.
    
    # 랜덤 시드를 설정하여 재현성을 보장합니다.
    os.environ['PYTHONHASHSEED']=str(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    
    # 재현성을 위해 텐서플로우에서 사용하는 스레드 수를 1로 제한
    # tf.config.threading.set_inter_op_parallelism_threads(1)
    # tf.config.threading.set_intra_op_parallelism_threads(1)
    
    tf.random.set_seed(1)  # 텐서플로우 랜덤 시드 설정
    
    train_dir = "./TensorBoard/"+differentiation_str  # 텐서보드 로그 저장 경로
    # summary_writer = tf.summary.create_file_writer(train_dir)  # 텐서보드 요약 작성자
    listofDemands = [8, 32, 64]  # 요구사항 리스트
    copy_weights_interval = 50  # 가중치 복사 간격
    evaluation_interval = 20  # 평가 간격
    epsilon_start_decay = 70  # 엡실론 감소 시작 시점
    
    hparams = {  # 하이퍼파라미터 설정
        'l2': 0.1,
        'dropout_rate': 0.01,
        'link_state_dim': 20,
        'readout_units': 35,
        'learning_rate': 0.0001,
        'batch_size': 32,
        'T': 4, 
        'num_demands': len(listofDemands)
    }
    
    MAX_QUEUE_SIZE = 4000  # 최대 큐 크기
    ```
    
    ```python
    def cummax(alist, extractor):
        with tf.name_scope('cummax'):
            maxes = [tf.reduce_max(extractor(v)) + 1 for v in alist]
            cummaxes = [tf.zeros_like(maxes[0])]
            for i in range(len(maxes) - 1):
                cummaxes.append(tf.math.add_n(maxes[0:i + 1]))
        return cummaxes
    
    class DQNAgent:
        def __init__(self, batch_size):
            self.memory = deque(maxlen=MAX_QUEUE_SIZE)
            self.gamma = 0.95  # discount rate
            self.epsilon = 1.0 # exploration rate
            self.epsilon_min = 0.01
            self.epsilon_decay = 0.995
            self.writer = None
            self.K = 4 # K-paths
            self.listQValues = None
            self.numbersamples = batch_size
            self.action = None
            self.capacity_feature = None
            self.bw_allocated_feature = np.zeros((env_training.numEdges,len(env_training.listofDemands)))
    
            self.global_step = 0
            self.primary_network = gnn.myModel(hparams)
            self.primary_network.build()
            self.target_network = gnn.myModel(hparams)
            self.target_network.build()
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=hparams['learning_rate'],momentum=0.9,nesterov=True)
    
        def act(self, env, state, demand, source, destination, flagEvaluation):
            """
            Given a demand stored in the environment it allocates the K=4 shortest paths on the current 'state'
            and predicts the q_values of the K=4 different new graph states by using the GNN model.
            Picks the state according to epsilon-greedy approach. The flag=TRUE indicates that we are testing
            the model and thus, it won't activate the drop layers.
            """
            # Set to True if we need to compute K=4 q-values and take the maxium
            takeMax_epsilon = False
            # List of graphs
            listGraphs = []
            # List of graph features that are used in the cummax() call
            list_k_features = list()
            # Initialize action
            action = 0
    
            # We get the K-paths between source-destination
            pathList = env.allPaths[str(source) +':'+ str(destination)]
            path = 0
    
            # 1. Implement epsilon-greedy to pick allocation
            # If flagEvaluation==TRUE we are EVALUATING => take always the action that the agent is saying has higher q-value
            # Otherwise, we are training with normal epsilon-greedy strategy
            if flagEvaluation:
                # If evaluation, compute K=4 q-values and take the maxium value
                takeMax_epsilon = True
            else:
                # If training, compute epsilon-greedy
                z = np.random.random()
                if z > self.epsilon:
                    # Compute K=4 q-values and pick the one with highest value
                    # In case of multiple same max values, return the first one
                    takeMax_epsilon = True
                else:
                    # Pick a random path and compute only one q-value
                    path = np.random.randint(0, len(pathList))
                    action = path
    
            # 2. Allocate (S,D, linkDemand) demand using the K shortest paths
            while path < len(pathList):
                state_copy = np.copy(state)
                currentPath = pathList[path]
                i = 0
                j = 1
    
                # 3. Iterate over paths' pairs of nodes and allocate demand to bw_allocated
                while (j < len(currentPath)):
                    state_copy[env.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][1] = demand
                    i = i + 1
                    j = j + 1
    
                # 4. Add allocated graphs' features to the list. Later we will compute their q-values using cummax
                listGraphs.append(state_copy)
                features = self.get_graph_features(env, state_copy)
                list_k_features.append(features)
    
                if not takeMax_epsilon:
                    # If we don't need to compute the K=4 q-values we exit
                    break
    
                path = path + 1
    
            vs = [v for v in list_k_features]
    
            # We compute the graphs_ids to later perform the unsorted_segment_sum for each graph and obtain the 
            # link hidden states for each graph.
            graph_ids = [tf.fill([tf.shape(vs[it]['link_state'])[0]], it) for it in range(len(list_k_features))]
            first_offset = cummax(vs, lambda v: v['first'])
            second_offset = cummax(vs, lambda v: v['second'])
    
            tensors = ({
                'graph_id': tf.concat([v for v in graph_ids], axis=0),
                'link_state': tf.concat([v['link_state'] for v in vs], axis=0),
                'first': tf.concat([v['first'] + m for v, m in zip(vs, first_offset)], axis=0),
                'second': tf.concat([v['second'] + m for v, m in zip(vs, second_offset)], axis=0),
                'num_edges': tf.math.add_n([v['num_edges'] for v in vs]),
                }
            )        
    
            # Predict qvalues for all graphs within tensors
            self.listQValues = self.primary_network(tensors['link_state'], tensors['graph_id'], tensors['first'],
                            tensors['second'], tensors['num_edges'], training=False).numpy()
    
            if takeMax_epsilon:
                # We take the path with highest q-value
                action = np.argmax(self.listQValues)
            else:
                return path, list_k_features[0]
    
            return action, list_k_features[action]
        
        def get_graph_features(self, env, copyGraph):
            """
            We iterate over the converted graph nodes and take the features. The capacity and bw allocated features
            are normalized on the fly.
            """
            self.bw_allocated_feature.fill(0.0)
            # Normalize capacity feature
            self.capacity_feature = (copyGraph[:,0] - 100.00000001) / 200.0
    
            iter = 0
            for i in copyGraph[:, 1]:
                if i == 8:
                    self.bw_allocated_feature[iter][0] = 1
                elif i == 32:
                    self.bw_allocated_feature[iter][1] = 1
                elif i == 64:
                    self.bw_allocated_feature[iter][2] = 1
                iter = iter + 1
            
            sample = {
                'num_edges': env.numEdges,  
                'length': env.firstTrueSize,
                'betweenness': tf.convert_to_tensor(value=env.between_feature, dtype=tf.float32),
                'bw_allocated': tf.convert_to_tensor(value=self.bw_allocated_feature, dtype=tf.float32),
                'capacities': tf.convert_to_tensor(value=self.capacity_feature, dtype=tf.float32),
                'first': tf.convert_to_tensor(env.first, dtype=tf.int32),
                'second': tf.convert_to_tensor(env.second, dtype=tf.int32)
            }
    
            sample['capacities'] = tf.reshape(sample['capacities'][0:sample['num_edges']], [sample['num_edges'], 1])
            sample['betweenness'] = tf.reshape(sample['betweenness'][0:sample['num_edges']], [sample['num_edges'], 1])
    
            hiddenStates = tf.concat([sample['capacities'], sample['betweenness'], sample['bw_allocated']], axis=1)
    
            paddings = tf.constant([[0, 0], [0, hparams['link_state_dim'] - 2 - hparams['num_demands']]])
            link_state = tf.pad(tensor=hiddenStates, paddings=paddings, mode="CONSTANT")
    
            inputs = {'link_state': link_state, 'first': sample['first'][0:sample['length']],
                      'second': sample['second'][0:sample['length']], 'num_edges': sample['num_edges']}
    
            return inputs
        
        def _write_tf_summary(self, gradients, loss):
            with summary_writer.as_default():
                tf.summary.scalar(name="loss", data=loss[0], step=self.global_step)
                tf.summary.histogram(name='gradients_5', data=gradients[5], step=self.global_step)
                tf.summary.histogram(name='gradients_7', data=gradients[7], step=self.global_step)
                tf.summary.histogram(name='gradients_9', data=gradients[9], step=self.global_step)
                tf.summary.histogram(name='FirstLayer/kernel:0', data=self.primary_network.variables[0], step=self.global_step)
                tf.summary.histogram(name='FirstLayer/bias:0', data=self.primary_network.variables[1], step=self.global_step)
                tf.summary.histogram(name='kernel:0', data=self.primary_network.variables[2], step=self.global_step)
                tf.summary.histogram(name='recurrent_kernel:0', data=self.primary_network.variables[3], step=self.global_step)
                tf.summary.histogram(name='bias:0', data=self.primary_network.variables[4], step=self.global_step)
                tf.summary.histogram(name='Readout1/kernel:0', data=self.primary_network.variables[5], step=self.global_step)
                tf.summary.histogram(name='Readout1/bias:0', data=self.primary_network.variables[6], step=self.global_step)
                tf.summary.histogram(name='Readout2/kernel:0', data=self.primary_network.variables[7], step=self.global_step)
                tf.summary.histogram(name='Readout2/bias:0', data=self.primary_network.variables[8], step=self.global_step)
                tf.summary.histogram(name='Readout3/kernel:0', data=self.primary_network.variables[9], step=self.global_step)
                tf.summary.histogram(name='Readout3/bias:0', data=self.primary_network.variables[10], step=self.global_step)
                summary_writer.flush()
                self.global_step = self.global_step + 1
    
        @tf.function
        def _forward_pass(self, x):
            prediction_state = self.primary_network(x[0], x[1], x[2], x[3], x[4], training=True)
            preds_next_target = tf.stop_gradient(self.target_network(x[6], x[7], x[9], x[10], x[11], training=True))
            return prediction_state, preds_next_target
    
        def _train_step(self, batch):
            # Record operations for automatic differentiation
            with tf.GradientTape() as tape:
                preds_state = []
                target = []
                for x in batch:
                    prediction_state, preds_next_target = self._forward_pass(x)
                    # Take q-value of the action performed
                    preds_state.append(prediction_state[0])
                    # We multiple by 0 if done==TRUE to cancel the second term
                    target.append(tf.stop_gradient([x[5] + self.gamma*tf.math.reduce_max(preds_next_target)*(1-x[8])]))
    
                loss = tf.keras.losses.MSE(tf.stack(target, axis=1), tf.stack(preds_state, axis=1))
                # Loss function using L2 Regularization
                regularization_loss = sum(self.primary_network.losses)
                loss = loss + regularization_loss
    
            # Computes the gradient using operations recorded in context of this tape
            grad = tape.gradient(loss, self.primary_network.variables)
            #gradients, _ = tf.clip_by_global_norm(grad, 5.0)
            gradients = [tf.clip_by_value(gradient, -1., 1.) for gradient in grad]
            self.optimizer.apply_gradients(zip(gradients, self.primary_network.variables))
            del tape
            return grad, loss
        
        def replay(self, episode):
            for i in range(MULTI_FACTOR_BATCH):
                batch = random.sample(self.memory, self.numbersamples)
                
                grad, loss = self._train_step(batch)
                if i%store_loss==0:
                    fileLogs.write(".," + '%.9f' % loss.numpy() + ",\n")
            
            # Soft weights update
            # for t, e in zip(self.target_network.trainable_variables, self.primary_network.trainable_variables):
            #     t.assign(t * (1 - TAU) + e * TAU)
    
            # Hard weights update
            if episode % copy_weights_interval == 0:
                self.target_network.set_weights(self.primary_network.get_weights()) 
            # if episode % evaluation_interval == 0:
            #     self._write_tf_summary(grad, loss)
            gc.collect()
        
        def add_sample(self, env_training, state_action, action, reward, done, new_state, new_demand, new_source, new_destination):
            self.bw_allocated_feature.fill(0.0)
            new_state_copy = np.copy(new_state)
    
            state_action['graph_id'] = tf.fill([tf.shape(state_action['link_state'])[0]], 0)
        
            # We get the K-paths between new_source-new_destination
            pathList = env_training.allPaths[str(new_source) +':'+ str(new_destination)]
            path = 0
            list_k_features = list()
    
            # 2. Allocate (S,D, linkDemand) demand using the K shortest paths
            while path < len(pathList):
                currentPath = pathList[path]
                i = 0
                j = 1
    
                # 3. Iterate over paths' pairs of nodes and allocate new_demand to bw_allocated
                while (j < len(currentPath)):
                    new_state_copy[env_training.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][1] = new_demand
                    i = i + 1
                    j = j + 1
    
                # 4. Add allocated graphs' features to the list. Later we will compute it's qvalues using cummax
                features = agent.get_graph_features(env_training, new_state_copy)
    
                list_k_features.append(features)
                path = path + 1
                new_state_copy[:,1] = 0
            
            vs = [v for v in list_k_features]
    
            # We compute the graphs_ids to later perform the unsorted_segment_sum for each graph and obtain the 
            # link hidden states for each graph.
            graph_ids = [tf.fill([tf.shape(vs[it]['link_state'])[0]], it) for it in range(len(list_k_features))]
            first_offset = cummax(vs, lambda v: v['first'])
            second_offset = cummax(vs, lambda v: v['second'])
    
            tensors = ({
                    'graph_id': tf.concat([v for v in graph_ids], axis=0),
                    'link_state': tf.concat([v['link_state'] for v in vs], axis=0),
                    'first': tf.concat([v['first'] + m for v, m in zip(vs, first_offset)], axis=0),
                    'second': tf.concat([v['second'] + m for v, m in zip(vs, second_offset)], axis=0),
                    'num_edges': tf.math.add_n([v['num_edges'] for v in vs]),
                }
            )    
            
            # We store the state with the action marked, the graph ids, first, second, num_edges, the reward, 
            # new_state(-1 because we don't need it in this case), the graph ids, done, first, second, number of edges
            self.memory.append((state_action['link_state'], state_action['graph_id'], state_action['first'], # 2
                            state_action['second'], tf.convert_to_tensor(state_action['num_edges']), # 4
                            tf.convert_to_tensor(reward, dtype=tf.float32), tensors['link_state'], tensors['graph_id'], # 7
                            tf.convert_to_tensor(int(done==True), dtype=tf.float32), tensors['first'], tensors['second'], # 10 
                            tf.convert_to_tensor(tensors['num_edges']))) # 12
    
    if __name__ == "__main__":
        # python train_DQN.py
        # Get the environment and extract the number of actions.
        env_training = gym.make(ENV_NAME)
        np.random.seed(SEED)
        env_training.seed(SEED)
        env_training.generate_environment(graph_topology, listofDemands)
    
        env_eval = gym.make(ENV_NAME)
        np.random.seed(SEED)
        env_eval.seed(SEED)
        env_eval.generate_environment(graph_topology, listofDemands)
    
        batch_size = hparams['batch_size']
        agent = DQNAgent(batch_size)
    
        eval_ep = 0
        train_ep = 0
        max_reward = 0
        reward_id = 0
    
        if not os.path.exists("./Logs"):
            os.makedirs("./Logs")
    
        # We store all the information in a Log file and later we parse this file 
        # to extract all the relevant information
        fileLogs = open("./Logs/exp" + differentiation_str + "Logs.txt", "a")
    
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    
        checkpoint = tf.train.Checkpoint(model=agent.primary_network, optimizer=agent.optimizer)
    
        rewards_test = np.zeros(EVALUATION_EPISODES)
    
        for eps in range(EVALUATION_EPISODES):
            state, demand, source, destination = env_eval.reset()
            rewardAddTest = 0
            while 1:
                # We execute evaluation over current state
                # demand, src, dst
                action, _ = agent.act(env_eval, state, demand, source, destination, True)
                
                new_state, reward, done, demand, source, destination = env_eval.make_step(state, action, demand, source, destination)
                rewardAddTest = rewardAddTest + reward
                state = new_state
                if done:
                    break
            rewards_test[eps] = rewardAddTest
    
        evalMeanReward = np.mean(rewards_test)
        fileLogs.write(">," + str(evalMeanReward) + ",\n")
        fileLogs.write("-," + str(agent.epsilon) + ",\n")
        fileLogs.flush()
    
        counter_store_model = 1
    
        for ep_it in range(ITERATIONS):
            if ep_it%5==0:
                print("Training iteration: ", ep_it)
    
            if ep_it==0:
                # At the beginning we don't have any experiences in the buffer. Thus, we force to
                # perform more training episodes than usually
                train_episodes = FIRST_WORK_TRAIN_EPISODE
            else:
                train_episodes = TRAINING_EPISODES
            for _ in range(train_episodes):
                # Used to clean the TF cache
                tf.random.set_seed(1)
                
                state, demand, source, destination = env_training.reset()            
    
                while 1:
                    # We execute evaluation over current state
                    action, state_action = agent.act(env_training, state, demand, source, destination, False)
                    new_state, reward, done, new_demand, new_source, new_destination = env_training.make_step(state, action, demand, source, destination)
    
                    agent.add_sample(env_training, state_action, action, reward, done, new_state, new_demand, new_source, new_destination)
                    state = new_state
                    demand = new_demand
                    source = new_source
                    destination = new_destination
                    if done:
                        break
    
            agent.replay(ep_it)
    
            # Decrease epsilon (from epsion-greedy exploration strategy)
            if ep_it > epsilon_start_decay and agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay
                agent.epsilon *= agent.epsilon_decay
    
            # We only evaluate the model every evaluation_interval steps
            if ep_it % evaluation_interval == 0:
                for eps in range(EVALUATION_EPISODES):
                    state, demand, source, destination = env_eval.reset()
                    rewardAddTest = 0
                    while 1:
                        # We execute evaluation over current state
                        action, _ = agent.act(env_eval, state, demand, source, destination, True)
                        
                        new_state, reward, done, demand, source, destination = env_eval.make_step(state, action, demand, source, destination)
                        rewardAddTest = rewardAddTest + reward
                        state = new_state
                        if done:
                            break
                    rewards_test[eps] = rewardAddTest
                evalMeanReward = np.mean(rewards_test)
    
                if evalMeanReward>max_reward:
                    max_reward = evalMeanReward
                    reward_id = counter_store_model
    
                fileLogs.write(">," + str(evalMeanReward) + ",\n")
                fileLogs.write("-," + str(agent.epsilon) + ",\n")
    
                # Store trained model
                checkpoint.save(checkpoint_prefix)
                fileLogs.write("MAX REWD: " + str(max_reward) + " MODEL_ID: " + str(reward_id) +",\n")
                counter_store_model = counter_store_model + 1
    
            fileLogs.flush()
    
            # Invoke garbage collection
            # tf.keras.backend.clear_session()
            gc.collect()
        
        for eps in range(EVALUATION_EPISODES):
            state, demand, source, destination = env_eval.reset()
            rewardAddTest = 0
            while 1:
                # We execute evaluation over current state
                # demand, src, dst
                action, _ = agent.act(env_eval, state, demand, source, destination, True)
                
                new_state, reward, done, demand, source, destination = env_eval.make_step(state, action, demand, source, destination)
                rewardAddTest = rewardAddTest + reward
                state = new_state
                if done:
                    break
            rewards_test[eps] = rewardAddTest
        evalMeanReward = np.mean(rewards_test)
    
        if evalMeanReward>max_reward:
            max_reward = evalMeanReward
            reward_id = counter_store_model
    
        fileLogs.write(">," + str(evalMeanReward) + ",\n")
        fileLogs.write("-," + str(agent.epsilon) + ",\n")
    
        # Store trained model
        checkpoint.save(checkpoint_prefix)
        fileLogs.write("MAX REWD: " + str(max_reward) + " MODEL_ID: " + str(reward_id) +",\n")
        
        fileLogs.flush()
        fileLogs.close()
    ```