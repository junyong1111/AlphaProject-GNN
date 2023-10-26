# Light GCN 심층

| 지표 (Metric) | 설명 (Description) |
| --- | --- |
| Loss | 모델이 훈련 중에나 테스트 중에 계산한 손실 함수의 값. 낮은 값이 더 좋은 모델을 의미한다. |
| ROC AUC | 이진 분류에서 모델의 성능을 평가하는 지표. 값이 1에 가까울수록 좋은 성능을, 0.5에 가까울수록 무작위 추측 수준의 성능을 의미한다. |
| Precision | 모델이 추천한 아이템 중 실제로 사용자가 관심을 가진 아이템의 비율.
예: 100개의 아이템을 추천했는데, 그 중 8개만 사용자가 실제로 좋아한다면 precision은 0.08 |
| Recall | 사용자가 관심을 가진 모든 아이템 중에서 모델이 얼마나 많은 아이템을 올바르게 추천했는지의 비율.
예: 사용자가 100개의 아이템을 좋아하는데, 모델이 그 중 7.6개를 올바르게 추천했다면 recall은 0.076 |
| NDCG | 추천된 아이템의 순서를 고려한 평가 지표. 아이템의 순서가 사용자의 선호도와 얼마나 잘 일치하는지를 나타낸다. |

### 1번 USER가 평가한 4~5점 데이터 30개를 제거 후 얼마나 추천하는지 확인

### 라이브러리, 블로그 코드 모델 코드 분석

### **embed_size  32 고정 epochs 20000~ 40000** 6개

### Case1

- Blog

| test_loss | 86.71565 |
| --- | --- |
| test_recall@10 | 0.0961 |
| test_precision@10 | 0.05709 |
| test_ndcg@10 | 0.09601 |

![1](https://github.com/junyong1111/AlphaProject-GNN/assets/79856225/342642f3-2411-4228-877f-521b4f49844b)

```python
title: Shawshank Redemption, The (1994), genres: Crime|Drama 
title: Pulp Fiction (1994), genres: Comedy|Crime|Drama|Thriller 
**title: Forrest Gump (1994), genres: Comedy|Drama|Romance|War** 
**title: Star Wars: Episode IV - A New Hope (1977), genres: Action|Adventure|Sci-Fi** 
**title: Silence of the Lambs, The (1991), genres: Crime|Horror|Thriller** 
**title: Schindler's List (1993), genres: Drama|War** 
**title: Usual Suspects, The (1995), genres: Crime|Mystery|Thriller** 
title: Godfather, The (1972), genres: Crime|Drama 
title: Terminator 2: Judgment Day (1991), genres: Action|Sci-Fi 
**title: Braveheart (1995), genres: Action|Drama|War** 
==========================BLOG LightGCN=============================
```

```python
title: ['Toy Story (1995)'], genres: ['Adventure|Animation|Children|Comedy|Fantasy'] 
title: ['Grumpier Old Men (1995)'], genres: ['Comedy|Romance'] 
title: ['Heat (1995)'], genres: ['Action|Crime|Thriller'] 
title: ['Seven (a.k.a. Se7en) (1995)'], genres: ['Mystery|Thriller'] 
**title: ['Usual Suspects, The (1995)'], genres: ['Crime|Mystery|Thriller']** 
title: ['Bottle Rocket (1996)'], genres: ['Adventure|Comedy|Crime|Romance'] 
**title: ['Braveheart (1995)'], genres: ['Action|Drama|War']** 
title: ['Rob Roy (1995)'], genres: ['Action|Drama|Romance|War'] 
title: ['Canadian Bacon (1995)'], genres: ['Comedy|War'] 
title: ['Desperado (1995)'], genres: ['Action|Romance|Western'] 
title: ['Billy Madison (1995)'], genres: ['Comedy'] 
title: ['Dumb & Dumber (Dumb and Dumber) (1994)'], genres: ['Adventure|Comedy'] 
title: ['Ed Wood (1994)'], genres: ['Comedy|Drama'] 
**title: ['Star Wars: Episode IV - A New Hope (1977)'], genres: ['Action|Adventure|Sci-Fi']** 
title: ['Tommy Boy (1995)'], genres: ['Comedy'] 
title: ['Clear and Present Danger (1994)'], genres: ['Action|Crime|Drama|Thriller'] 
**title: ['Forrest Gump (1994)'], genres: ['Comedy|Drama|Romance|War']** 
title: ['Jungle Book, The (1994)'], genres: ['Adventure|Children|Romance'] 
title: ['Mask, The (1994)'], genres: ['Action|Comedy|Crime|Fantasy'] 
title: ['Dazed and Confused (1993)'], genres: ['Comedy'] 
title: ['Fugitive, The (1993)'], genres: ['Thriller'] 
title: ['Jurassic Park (1993)'], genres: ['Action|Adventure|Sci-Fi|Thriller'] 
**title: ["Schindler's List (1993)"], genres: ['Drama|War']** 
title: ['So I Married an Axe Murderer (1993)'], genres: ['Comedy|Romance|Thriller'] 
title: ['Three Musketeers, The (1993)'], genres: ['Action|Adventure|Comedy|Romance'] 
title: ['Tombstone (1993)'], genres: ['Action|Drama|Western'] 
title: ['Dances with Wolves (1990)'], genres: ['Adventure|Drama|Western'] 
title: ['Batman (1989)'], genres: ['Action|Crime|Thriller'] 
**title: ['Silence of the Lambs, The (1991)'], genres: ['Crime|Horror|Thriller']** 
title: ['Pinocchio (1940)'], genres: ['Animation|Children|Fantasy|Musical']
```

```python
**embed_size = 32
n_epochs = 20000
user_id = 1
excluded_user = 1
rating_threshold = 4
K = 10**
```

### Case2

- Blog

| test_loss | 86.71565 |
| --- | --- |
| test_recall@10 | 0.0961 |
| test_precision@10 | 0.05709 |
| test_ndcg@10 | 0.09601 |

![2](https://github.com/junyong1111/AlphaProject-GNN/assets/79856225/125d29ca-7538-483b-a92f-f442cb1f444e)

```python
title: Shawshank Redemption, The (1994), genres: Crime|Drama 
**title: Forrest Gump (1994), genres: Comedy|Drama|Romance|War** 
title: Pulp Fiction (1994), genres: Comedy|Crime|Drama|Thriller 
**title: Silence of the Lambs, The (1991), genres: Crime|Horror|Thriller** 
**title: Star Wars: Episode IV - A New Hope (1977), genres: Action|Adventure|Sci-Fi** 
**title: Braveheart (1995), genres: Action|Drama|War** 
title: Godfather, The (1972), genres: Crime|Drama 
**title: Schindler's List (1993), genres: Drama|War** 
title: Terminator 2: Judgment Day (1991), genres: Action|Sci-Fi 
**title: Usual Suspects, The (1995), genres: Crime|Mystery|Thriller** 
==========================BLOG LightGCN=============================
```

```python
title: ['Toy Story (1995)'], genres: ['Adventure|Animation|Children|Comedy|Fantasy'] 
title: ['Grumpier Old Men (1995)'], genres: ['Comedy|Romance'] 
title: ['Heat (1995)'], genres: ['Action|Crime|Thriller'] 
title: ['Seven (a.k.a. Se7en) (1995)'], genres: ['Mystery|Thriller'] 
**title: ['Usual Suspects, The (1995)'], genres: ['Crime|Mystery|Thriller']** 
title: ['Bottle Rocket (1996)'], genres: ['Adventure|Comedy|Crime|Romance'] 
**title: ['Braveheart (1995)'], genres: ['Action|Drama|War']** 
title: ['Rob Roy (1995)'], genres: ['Action|Drama|Romance|War'] 
title: ['Canadian Bacon (1995)'], genres: ['Comedy|War'] 
title: ['Desperado (1995)'], genres: ['Action|Romance|Western'] 
title: ['Billy Madison (1995)'], genres: ['Comedy'] 
title: ['Dumb & Dumber (Dumb and Dumber) (1994)'], genres: ['Adventure|Comedy'] 
title: ['Ed Wood (1994)'], genres: ['Comedy|Drama'] 
**title: ['Star Wars: Episode IV - A New Hope (1977)'], genres: ['Action|Adventure|Sci-Fi']** 
title: ['Tommy Boy (1995)'], genres: ['Comedy'] 
title: ['Clear and Present Danger (1994)'], genres: ['Action|Crime|Drama|Thriller'] 
**title: ['Forrest Gump (1994)'], genres: ['Comedy|Drama|Romance|War']** 
title: ['Jungle Book, The (1994)'], genres: ['Adventure|Children|Romance'] 
title: ['Mask, The (1994)'], genres: ['Action|Comedy|Crime|Fantasy'] 
title: ['Dazed and Confused (1993)'], genres: ['Comedy'] 
title: ['Fugitive, The (1993)'], genres: ['Thriller'] 
title: ['Jurassic Park (1993)'], genres: ['Action|Adventure|Sci-Fi|Thriller'] 
**title: ["Schindler's List (1993)"], genres: ['Drama|War']** 
title: ['So I Married an Axe Murderer (1993)'], genres: ['Comedy|Romance|Thriller'] 
title: ['Three Musketeers, The (1993)'], genres: ['Action|Adventure|Comedy|Romance'] 
title: ['Tombstone (1993)'], genres: ['Action|Drama|Western'] 
title: ['Dances with Wolves (1990)'], genres: ['Adventure|Drama|Western'] 
title: ['Batman (1989)'], genres: ['Action|Crime|Thriller'] 
**title: ['Silence of the Lambs, The (1991)'], genres: ['Crime|Horror|Thriller']** 
title: ['Pinocchio (1940)'], genres: ['Animation|Children|Fantasy|Musical']
```

```python
**embed_size = 32
n_epochs = 30000
user_id = 1
excluded_user = 1
rating_threshold = 4
K = 10**
```

### Case3

- Blog

| test_loss | 92.59604 |
| --- | --- |
| test_recall@10 | 0.1083 |
| test_precision@10 | 0.06649 |
| test_ndcg@10 | 0.11069 |

![3](https://github.com/junyong1111/AlphaProject-GNN/assets/79856225/f0a632a7-e5a3-4ae5-8c7e-df865b8b47a7)

```python
title: Shawshank Redemption, The (1994), genres: Crime|Drama 
title: Pulp Fiction (1994), genres: Comedy|Crime|Drama|Thriller 
**title: Forrest Gump (1994), genres: Comedy|Drama|Romance|War** 
**title: Silence of the Lambs, The (1991), genres: Crime|Horror|Thriller** 
**title: Star Wars: Episode IV - A New Hope (1977), genres: Action|Adventure|Sci-Fi** 
**title: Schindler's List (1993), genres: Drama|War** 
title: Godfather, The (1972), genres: Crime|Drama 
**title: Braveheart (1995), genres: Action|Drama|War** 
title: Terminator 2: Judgment Day (1991), genres: Action|Sci-Fi 
**title: Usual Suspects, The (1995), genres: Crime|Mystery|Thriller** 
==========================BLOG LightGCN=============================
```

```python
title: ['Toy Story (1995)'], genres: ['Adventure|Animation|Children|Comedy|Fantasy'] 
title: ['Grumpier Old Men (1995)'], genres: ['Comedy|Romance'] 
title: ['Heat (1995)'], genres: ['Action|Crime|Thriller'] 
title: ['Seven (a.k.a. Se7en) (1995)'], genres: ['Mystery|Thriller'] 
**title: ['Usual Suspects, The (1995)'], genres: ['Crime|Mystery|Thriller']** 
title: ['Bottle Rocket (1996)'], genres: ['Adventure|Comedy|Crime|Romance'] 
**title: ['Braveheart (1995)'], genres: ['Action|Drama|War']** 
title: ['Rob Roy (1995)'], genres: ['Action|Drama|Romance|War'] 
title: ['Canadian Bacon (1995)'], genres: ['Comedy|War'] 
title: ['Desperado (1995)'], genres: ['Action|Romance|Western'] 
title: ['Billy Madison (1995)'], genres: ['Comedy'] 
title: ['Dumb & Dumber (Dumb and Dumber) (1994)'], genres: ['Adventure|Comedy'] 
title: ['Ed Wood (1994)'], genres: ['Comedy|Drama'] 
**title: ['Star Wars: Episode IV - A New Hope (1977)'], genres: ['Action|Adventure|Sci-Fi']** 
title: ['Tommy Boy (1995)'], genres: ['Comedy'] 
title: ['Clear and Present Danger (1994)'], genres: ['Action|Crime|Drama|Thriller'] 
**title: ['Forrest Gump (1994)'], genres: ['Comedy|Drama|Romance|War']** 
title: ['Jungle Book, The (1994)'], genres: ['Adventure|Children|Romance'] 
title: ['Mask, The (1994)'], genres: ['Action|Comedy|Crime|Fantasy'] 
title: ['Dazed and Confused (1993)'], genres: ['Comedy'] 
title: ['Fugitive, The (1993)'], genres: ['Thriller'] 
title: ['Jurassic Park (1993)'], genres: ['Action|Adventure|Sci-Fi|Thriller'] 
**title: ["Schindler's List (1993)"], genres: ['Drama|War']** 
title: ['So I Married an Axe Murderer (1993)'], genres: ['Comedy|Romance|Thriller'] 
title: ['Three Musketeers, The (1993)'], genres: ['Action|Adventure|Comedy|Romance'] 
title: ['Tombstone (1993)'], genres: ['Action|Drama|Western'] 
title: ['Dances with Wolves (1990)'], genres: ['Adventure|Drama|Western'] 
title: ['Batman (1989)'], genres: ['Action|Crime|Thriller'] 
**title: ['Silence of the Lambs, The (1991)'], genres: ['Crime|Horror|Thriller']** 
title: ['Pinocchio (1940)'], genres: ['Animation|Children|Fantasy|Musical']
```

```python
**embed_size = 32
n_epochs = 40000
user_id = 1
excluded_user = 1
rating_threshold = 4
K = 10**
```

### **embed_size  64 고정 epochs 20000~ 40000** 6개

### Case1

![4](https://github.com/junyong1111/AlphaProject-GNN/assets/79856225/496a192e-c7ab-4b96-9a1f-7e170c103457)

- Blog

| test_loss | 179.53043 |
| --- | --- |
| test_recall@10 | 0.10605 |
| test_precision@10 | 0.06543 |
| test_ndcg@10 | 0.11047 |

```python
title: Shawshank Redemption, The (1994), genres: Crime|Drama 
**title: Forrest Gump (1994), genres: Comedy|Drama|Romance|War** 
title: Pulp Fiction (1994), genres: Comedy|Crime|Drama|Thriller 
**title: Silence of the Lambs, The (1991), genres: Crime|Horror|Thriller** 
**title: Star Wars: Episode IV - A New Hope (1977), genres: Action|Adventure|Sci-Fi** 
title: Godfather, The (1972), genres: Crime|Drama 
**title: Braveheart (1995), genres: Action|Drama|War** 
title: Terminator 2: Judgment Day (1991), genres: Action|Sci-Fi 
**title: Usual Suspects, The (1995), genres: Crime|Mystery|Thriller** 
**title: Schindler's List (1993), genres: Drama|War** 
==========================BLOG LightGCN=============================
```

```python
title: ['Toy Story (1995)'], genres: ['Adventure|Animation|Children|Comedy|Fantasy'] 
title: ['Grumpier Old Men (1995)'], genres: ['Comedy|Romance'] 
title: ['Heat (1995)'], genres: ['Action|Crime|Thriller'] 
title: ['Seven (a.k.a. Se7en) (1995)'], genres: ['Mystery|Thriller'] 
**title: ['Usual Suspects, The (1995)'], genres: ['Crime|Mystery|Thriller']** 
title: ['Bottle Rocket (1996)'], genres: ['Adventure|Comedy|Crime|Romance'] 
**title: ['Braveheart (1995)'], genres: ['Action|Drama|War']** 
title: ['Rob Roy (1995)'], genres: ['Action|Drama|Romance|War'] 
title: ['Canadian Bacon (1995)'], genres: ['Comedy|War'] 
title: ['Desperado (1995)'], genres: ['Action|Romance|Western'] 
title: ['Billy Madison (1995)'], genres: ['Comedy'] 
title: ['Dumb & Dumber (Dumb and Dumber) (1994)'], genres: ['Adventure|Comedy'] 
title: ['Ed Wood (1994)'], genres: ['Comedy|Drama'] 
**title: ['Star Wars: Episode IV - A New Hope (1977)'], genres: ['Action|Adventure|Sci-Fi']** 
title: ['Tommy Boy (1995)'], genres: ['Comedy'] 
title: ['Clear and Present Danger (1994)'], genres: ['Action|Crime|Drama|Thriller'] 
**title: ['Forrest Gump (1994)'], genres: ['Comedy|Drama|Romance|War']** 
title: ['Jungle Book, The (1994)'], genres: ['Adventure|Children|Romance'] 
title: ['Mask, The (1994)'], genres: ['Action|Comedy|Crime|Fantasy'] 
title: ['Dazed and Confused (1993)'], genres: ['Comedy'] 
title: ['Fugitive, The (1993)'], genres: ['Thriller'] 
title: ['Jurassic Park (1993)'], genres: ['Action|Adventure|Sci-Fi|Thriller'] 
**title: ["Schindler's List (1993)"], genres: ['Drama|War']** 
title: ['So I Married an Axe Murderer (1993)'], genres: ['Comedy|Romance|Thriller'] 
title: ['Three Musketeers, The (1993)'], genres: ['Action|Adventure|Comedy|Romance'] 
title: ['Tombstone (1993)'], genres: ['Action|Drama|Western'] 
title: ['Dances with Wolves (1990)'], genres: ['Adventure|Drama|Western'] 
title: ['Batman (1989)'], genres: ['Action|Crime|Thriller'] 
**title: ['Silence of the Lambs, The (1991)'], genres: ['Crime|Horror|Thriller']** 
title: ['Pinocchio (1940)'], genres: ['Animation|Children|Fantasy|Musical']
```

```python
**embed_size = 64
n_epochs = 20000
user_id = 1
excluded_user = 1
rating_threshold = 4
K = 10**
```

### Case2

- Blog

| test_loss | 184.10132 |
| --- | --- |
| test_recall@10 | 0.10546 |
| test_precision@10 | 0.06561 |
| test_ndcg@10 | 0.10823 |

![5](https://github.com/junyong1111/AlphaProject-GNN/assets/79856225/c3bd829b-b96f-4f25-ae6c-f8acf3f3b1a7)

```python

title: Forrest Gump (1994), genres: Comedy|Drama|Romance|War 
title: Shawshank Redemption, The (1994), genres: Crime|Drama 
title: Pulp Fiction (1994), genres: Comedy|Crime|Drama|Thriller 
title: Silence of the Lambs, The (1991), genres: Crime|Horror|Thriller 
title: Star Wars: Episode IV - A New Hope (1977), genres: Action|Adventure|Sci-Fi 
title: Braveheart (1995), genres: Action|Drama|War 
title: Godfather, The (1972), genres: Crime|Drama 
title: Schindler's List (1993), genres: Drama|War 
title: Usual Suspects, The (1995), genres: Crime|Mystery|Thriller 
title: Terminator 2: Judgment Day (1991), genres: Action|Sci-Fi 
==========================BLOG LightGCN=======================================================LIB LightGCN=============================
Forrest Gump (1994)
Silence of the Lambs, The (1991)
Star Wars: Episode IV - A New Hope (1977)
Braveheart (1995)
Schindler's List (1993)
Usual Suspects, The (1995)
==========================BLOG LightGCN=============================
```

```python
**embed_size = 64
n_epochs = 30000
user_id = 1
excluded_user = 1
rating_threshold = 4
K = 10**
```

### Case3

- Blog

| test_loss | 182.61363 |
| --- | --- |
| test_recall@10 | 0.10748 |
| test_precision@10 | 0.06631 |
| test_ndcg@10 | 0.11035 |

![6](https://github.com/junyong1111/AlphaProject-GNN/assets/79856225/c8a76916-9dcb-4d8f-8148-82c3e6ab8e97)

```python
title: Shawshank Redemption, The (1994), genres: Crime|Drama 
title: Forrest Gump (1994), genres: Comedy|Drama|Romance|War 
title: Pulp Fiction (1994), genres: Comedy|Crime|Drama|Thriller 
title: Silence of the Lambs, The (1991), genres: Crime|Horror|Thriller 
title: Star Wars: Episode IV - A New Hope (1977), genres: Action|Adventure|Sci-Fi 
title: Braveheart (1995), genres: Action|Drama|War 
title: Godfather, The (1972), genres: Crime|Drama 
title: Schindler's List (1993), genres: Drama|War 
title: Usual Suspects, The (1995), genres: Crime|Mystery|Thriller 
title: Terminator 2: Judgment Day (1991), genres: Action|Sci-Fi 
==========================BLOG LightGCN=============================
```

```python
Forrest Gump (1994)
Silence of the Lambs, The (1991)
Star Wars: Episode IV - A New Hope (1977)
Braveheart (1995)
Schindler's List (1993)
Usual Suspects, The (1995)
==========================BLOG LightGCN=============================
```

```python
**embed_size = 64
n_epochs = 40000
user_id = 1
excluded_user = 1
rating_threshold = 4
K = 10**
```

### **embed_size  128 고정 epochs 20000~ 40000 3 ~** 6개

### Case1

![7](https://github.com/junyong1111/AlphaProject-GNN/assets/79856225/e0152206-a0ff-4663-a889-6c99e377b358)

- Blog

| test_loss | 179.53043 |
| --- | --- |
| test_recall@10 | 0.10605 |
| test_precision@10 | 0.06543 |
| test_ndcg@10 | 0.11047 |

```python
title: Shawshank Redemption, The (1994), genres: Crime|Drama 
**title: Forrest Gump (1994), genres: Comedy|Drama|Romance|War** 
title: Pulp Fiction (1994), genres: Comedy|Crime|Drama|Thriller 
**title: Silence of the Lambs, The (1991), genres: Crime|Horror|Thriller** 
**title: Star Wars: Episode IV - A New Hope (1977), genres: Action|Adventure|Sci-Fi** 
title: Godfather, The (1972), genres: Crime|Drama 
**title: Braveheart (1995), genres: Action|Drama|War** 
title: Terminator 2: Judgment Day (1991), genres: Action|Sci-Fi 
**title: Usual Suspects, The (1995), genres: Crime|Mystery|Thriller** 
**title: Schindler's List (1993), genres: Drama|War** 
==========================BLOG LightGCN=============================
```

```python
title: ['Toy Story (1995)'], genres: ['Adventure|Animation|Children|Comedy|Fantasy'] 
title: ['Grumpier Old Men (1995)'], genres: ['Comedy|Romance'] 
title: ['Heat (1995)'], genres: ['Action|Crime|Thriller'] 
title: ['Seven (a.k.a. Se7en) (1995)'], genres: ['Mystery|Thriller'] 
**title: ['Usual Suspects, The (1995)'], genres: ['Crime|Mystery|Thriller']** 
title: ['Bottle Rocket (1996)'], genres: ['Adventure|Comedy|Crime|Romance'] 
**title: ['Braveheart (1995)'], genres: ['Action|Drama|War']** 
title: ['Rob Roy (1995)'], genres: ['Action|Drama|Romance|War'] 
title: ['Canadian Bacon (1995)'], genres: ['Comedy|War'] 
title: ['Desperado (1995)'], genres: ['Action|Romance|Western'] 
title: ['Billy Madison (1995)'], genres: ['Comedy'] 
title: ['Dumb & Dumber (Dumb and Dumber) (1994)'], genres: ['Adventure|Comedy'] 
title: ['Ed Wood (1994)'], genres: ['Comedy|Drama'] 
**title: ['Star Wars: Episode IV - A New Hope (1977)'], genres: ['Action|Adventure|Sci-Fi']** 
title: ['Tommy Boy (1995)'], genres: ['Comedy'] 
title: ['Clear and Present Danger (1994)'], genres: ['Action|Crime|Drama|Thriller'] 
**title: ['Forrest Gump (1994)'], genres: ['Comedy|Drama|Romance|War']** 
title: ['Jungle Book, The (1994)'], genres: ['Adventure|Children|Romance'] 
title: ['Mask, The (1994)'], genres: ['Action|Comedy|Crime|Fantasy'] 
title: ['Dazed and Confused (1993)'], genres: ['Comedy'] 
title: ['Fugitive, The (1993)'], genres: ['Thriller'] 
title: ['Jurassic Park (1993)'], genres: ['Action|Adventure|Sci-Fi|Thriller'] 
**title: ["Schindler's List (1993)"], genres: ['Drama|War']** 
title: ['So I Married an Axe Murderer (1993)'], genres: ['Comedy|Romance|Thriller'] 
title: ['Three Musketeers, The (1993)'], genres: ['Action|Adventure|Comedy|Romance'] 
title: ['Tombstone (1993)'], genres: ['Action|Drama|Western'] 
title: ['Dances with Wolves (1990)'], genres: ['Adventure|Drama|Western'] 
title: ['Batman (1989)'], genres: ['Action|Crime|Thriller'] 
**title: ['Silence of the Lambs, The (1991)'], genres: ['Crime|Horror|Thriller']** 
title: ['Pinocchio (1940)'], genres: ['Animation|Children|Fantasy|Musical']
```

```python
**embed_size = 64
n_epochs = 20000
user_id = 1
excluded_user = 1
rating_threshold = 4
K = 10**
```

### Case2

- Blog

| test_loss | 184.10132 |
| --- | --- |
| test_recall@10 | 0.10546 |
| test_precision@10 | 0.06561 |
| test_ndcg@10 | 0.10823 |

![8](https://github.com/junyong1111/AlphaProject-GNN/assets/79856225/e6ceb070-afb1-413e-b721-c28bdf994fe6)
```python

title: Forrest Gump (1994), genres: Comedy|Drama|Romance|War 
title: Shawshank Redemption, The (1994), genres: Crime|Drama 
title: Pulp Fiction (1994), genres: Comedy|Crime|Drama|Thriller 
title: Silence of the Lambs, The (1991), genres: Crime|Horror|Thriller 
title: Star Wars: Episode IV - A New Hope (1977), genres: Action|Adventure|Sci-Fi 
title: Braveheart (1995), genres: Action|Drama|War 
title: Godfather, The (1972), genres: Crime|Drama 
title: Schindler's List (1993), genres: Drama|War 
title: Usual Suspects, The (1995), genres: Crime|Mystery|Thriller 
title: Terminator 2: Judgment Day (1991), genres: Action|Sci-Fi 
==========================BLOG LightGCN=======================================================LIB LightGCN=============================
Forrest Gump (1994)
Silence of the Lambs, The (1991)
Star Wars: Episode IV - A New Hope (1977)
Braveheart (1995)
Schindler's List (1993)
Usual Suspects, The (1995)
==========================BLOG LightGCN=============================
```

```python
**embed_size = 64
n_epochs = 30000
user_id = 1
excluded_user = 1
rating_threshold = 4
K = 10**
```

### Case3

- Blog

| test_loss | 182.61363 |
| --- | --- |
| test_recall@10 | 0.10748 |
| test_precision@10 | 0.06631 |
| test_ndcg@10 | 0.11035 |

![9](https://github.com/junyong1111/AlphaProject-GNN/assets/79856225/4c9f03df-bb2c-4713-871d-c818f90c37f2)

```python
title: Shawshank Redemption, The (1994), genres: Crime|Drama 
title: Forrest Gump (1994), genres: Comedy|Drama|Romance|War 
title: Pulp Fiction (1994), genres: Comedy|Crime|Drama|Thriller 
title: Silence of the Lambs, The (1991), genres: Crime|Horror|Thriller 
title: Star Wars: Episode IV - A New Hope (1977), genres: Action|Adventure|Sci-Fi 
title: Braveheart (1995), genres: Action|Drama|War 
title: Godfather, The (1972), genres: Crime|Drama 
title: Schindler's List (1993), genres: Drama|War 
title: Usual Suspects, The (1995), genres: Crime|Mystery|Thriller 
title: Terminator 2: Judgment Day (1991), genres: Action|Sci-Fi 
==========================BLOG LightGCN=============================
```

```python
Forrest Gump (1994)
Silence of the Lambs, The (1991)
Star Wars: Episode IV - A New Hope (1977)
Braveheart (1995)
Schindler's List (1993)
Usual Suspects, The (1995)
==========================BLOG LightGCN=============================
```

```python
**embed_size = 64
n_epochs = 40000
user_id = 1
excluded_user = 1
rating_threshold = 4
K = 10**
```

### Case1

- Blog

| test_loss | 346.15338 |
| --- | --- |
| test_recall@10 | 0.10719 |
| test_precision@10 | 0.06526 |
| test_ndcg@10 | 0.11014 |

![10](https://github.com/junyong1111/AlphaProject-GNN/assets/79856225/38f15c3c-eb18-4841-8325-f47a093cf4e9)

```python
title: Shawshank Redemption, The (1994), genres: Crime|Drama 
title: Forrest Gump (1994), genres: Comedy|Drama|Romance|War 
title: Pulp Fiction (1994), genres: Comedy|Crime|Drama|Thriller 
title: Silence of the Lambs, The (1991), genres: Crime|Horror|Thriller 
title: Star Wars: Episode IV - A New Hope (1977), genres: Action|Adventure|Sci-Fi 
title: Godfather, The (1972), genres: Crime|Drama 
title: Braveheart (1995), genres: Action|Drama|War 
title: Usual Suspects, The (1995), genres: Crime|Mystery|Thriller 
title: Terminator 2: Judgment Day (1991), genres: Action|Sci-Fi 
title: Schindler's List (1993), genres: Drama|War 
==========================BLOG LightGCN=============================
```

```python
Forrest Gump (1994)
Silence of the Lambs, The (1991)
Star Wars: Episode IV - A New Hope (1977)
Braveheart (1995)
Usual Suspects, The (1995)
Schindler's List (1993)
==========================BLOG LightGCN=============================
```

```python
**embed_size = 128
n_epochs = 20000
user_id = 1
excluded_user = 1
rating_threshold = 4
K = 10**
```

### Case2

- Blog

| test_loss | 352.79431 |
| --- | --- |
| test_recall@10 | 0.09575 |
| test_precision@10 | 0.05639 |
| test_ndcg@10 | 0.09661 |

![11](https://github.com/junyong1111/AlphaProject-GNN/assets/79856225/cb8d9b7c-9aa6-4e7c-a587-b0286f54b3fb)

```python

title: Shawshank Redemption, The (1994), genres: Crime|Drama 
title: Forrest Gump (1994), genres: Comedy|Drama|Romance|War 
title: Pulp Fiction (1994), genres: Comedy|Crime|Drama|Thriller 
title: Silence of the Lambs, The (1991), genres: Crime|Horror|Thriller 
**title: Star Wars: Episode IV - A New Hope (1977), genres: Action|Adventure|Sci-Fi 
title: Schindler's List (1993), genres: Drama|War** 
**title: Usual Suspects, The (1995), genres: Crime|Mystery|Thriller** 
title: Godfather, The (1972), genres: Crime|Drama 
title: Braveheart (1995), genres: Action|Drama|War 
title: Terminator 2: Judgment Day (1991), genres: Action|Sci-Fi 
==========================BLOG LightGCN=============================title: ['Seven (a.k.a. Se7en) (1995)'], genres: ['Mystery|Thriller'] 
**title: ['Usual Suspects, The (1995)'], genres: ['Crime|Mystery|Thriller']** 
title: ['Bottle Rocket (1996)'], genres: ['Adventure|Comedy|Crime|Romance'] 
title: ['Rob Roy (1995)'], genres: ['Action|Drama|Romance|War'] 
title: ['Canadian Bacon (1995)'], genres: ['Comedy|War'] 
title: ['Desperado (1995)'], genres: ['Action|Romance|Western'] 
title: ['Billy Madison (1995)'], genres: ['Comedy'] 
title: ['Dumb & Dumber (Dumb and Dumber) (1994)'], genres: ['Adventure|Comedy'] 
**title: ['Star Wars: Episode IV - A New Hope (1977)'], genres: ['Action|Adventure|Sci-Fi']** 
title: ['Tommy Boy (1995)'], genres: ['Comedy'] 
title: ['Jungle Book, The (1994)'], genres: ['Adventure|Children|Romance'] 
title: ['Fugitive, The (1993)'], genres: ['Thriller'] 
**title: ["Schindler's List (1993)"], genres: ['Drama|War']** 
title: ['Tombstone (1993)'], genres: ['Action|Drama|Western'] 
title: ['Pinocchio (1940)'], genres: ['Animation|Children|Fantasy|Musical'] 
title: ['Fargo (1996)'], genres: ['Comedy|Crime|Drama|Thriller'] 
title: ['James and the Giant Peach (1996)'], genres: ['Adventure|Animation|Children|Fantasy|Musical'] 
title: ['Wizard of Oz, The (1939)'], genres: ['Adventure|Children|Fantasy|Musical'] 
title: ['Citizen Kane (1941)'], genres: ['Drama|Mystery'] 
title: ['Adventures of Robin Hood, The (1938)'], genres: ['Action|Adventure|Romance'] 
title: ['Mr. Smith Goes to Washington (1939)'], genres: ['Drama'] 
title: ['Winnie the Pooh and the Blustery Day (1968)'], genres: ['Animation|Children|Musical'] 
title: ['Three Caballeros, The (1945)'], genres: ['Animation|Children|Musical'] 
title: ['Sword in the Stone, The (1963)'], genres: ['Animation|Children|Fantasy|Musical'] 
title: ['Dumbo (1941)'], genres: ['Animation|Children|Drama|Musical'] 
title: ['Bedknobs and Broomsticks (1971)'], genres: ['Adventure|Children|Musical'] 
title: ['Alice in Wonderland (1951)'], genres: ['Adventure|Animation|Children|Fantasy|Musical'] 
title: ['Ghost and the Darkness, The (1996)'], genres: ['Action|Adventure'] 
title: ['Willy Wonka & the Chocolate Factory (1971)'], genres: ['Children|Comedy|Fantasy|Musical'] 
title: ["Monty Python's Life of Brian (1979)"], genres: ['Comedy']
```

```python
**embed_size = 128
n_epochs = 30000
user_id = 1
excluded_user = 1
rating_threshold = 4
K = 10**
```

### 라이브러리

- CODE
    
    [](https://github.com/massquantity/LibRecommender/blob/master/libreco/algorithms/lightgcn.py)
    
    ```python
    """Implementation of LightGCN."""
    import torch
    
    from .torch_modules import LightGCNModel
    from ..bases import EmbedBase, ModelMeta
    from ..torchops import device_config, set_torch_seed
    
    class LightGCN(EmbedBase, metaclass=ModelMeta, backend="torch"):
        """*LightGCN* 알고리즘.
    
        .. 주의::
            LightGCN은 ``랭킹`` 작업에서만 사용할 수 있습니다.
    
        파라미터
        ----------
        task : {'ranking'}
            추천 작업입니다. 참조: :ref:`Task`.
        data_info : :class:`~libreco.data.DataInfo` 객체
            학습 및 추론에 유용한 정보가 포함된 객체입니다.
        loss_type : {'cross_entropy', 'focal', 'bpr', 'max_margin'}, 기본값: 'bpr'
            모델 학습을 위한 손실.
        embed_size: int, 기본값: 16
            임베딩의 벡터 크기.
        n_epochs: int, 기본값: 10
            학습을 위한 에포크 수입니다.
        lr: float, 기본값 0.001
            학습을 위한 학습률.
        lr_decay : 부울, 기본값: False
            학습률 감쇠 사용 여부.
        엡실론 : float, 기본값: 1E-8
            아담 옵티마이저에서 수치 안정성을 향상시키기 위해 분모에 추가되는 작은 상수입니다.
            아담 옵티마이저.
        amsgrad : 부울, 기본값: False
            논문에서 AMSGrad 변형을 사용할지 여부.
            '아담과 그 너머의 융합 <https://openreview.net/forum?id=ryQu7f-RZ>`_.
        reg : float 또는 None, 기본값: None
            정규화 매개변수, 음수가 아니거나 None이어야 합니다.
        batch_size : int, 기본값: 256
            학습을 위한 배치 크기.
        num_neg : int, 기본값: 1
            각 양성 샘플에 대한 음성 샘플 수입니다.
        dropout_rate : float, 기본값: 0.0
            노드가 탈락할 확률. 0.0은 드롭아웃이 사용되지 않음을 의미합니다.
        n_layers : int, 기본값: 3
            GCN 레이어 수입니다.
        마진 : 플로트, 기본값 : 1.0
            max_margin` 손실에 사용되는 마진.
        sampler : {'random', 'unconsumed', 'popular'}, 기본값: 'random'
            네거티브 샘플링 전략.
            - 'random'은 무작위 샘플링을 의미합니다.
            - '미소비'는 대상 사용자가 이전에 소비하지 않은 항목을 샘플링합니다.
            - '인기'는 인기 있는 아이템을 네거티브 샘플로 샘플링할 확률이 높습니다.
    
        시드 : INT, 기본값 : 42
            임의의 시드입니다.
        device : {'cpu', 'cuda'}, 기본값: 'cuda'
            'torch.device <https://pytorch.org/docs/stable/tensor_attributes.html#torch.device>`_'를 참고하세요.
    
            .. 버전 변경:: 1.0.0
               '``torch.device(...)`` 대신 ``'cpu'`` 또는 ``'cuda'`` 타입을 허용합니다.
    
        lower_upper_bound : 튜플 또는 None, 기본값: None
            평가` 작업에 대한 하한 및 상한 점수 바운드.
        """
    		# 초기화 메서드 정의
        def __init__(
            self,
            task,
            data_info,
            loss_type="bpr",
            embed_size=16,
            n_epochs=20,
            lr=0.001,
            lr_decay=False,
            epsilon=1e-8,
            amsgrad=False,
            reg=None,
            batch_size=256,
            num_neg=1,
            dropout_rate=0.0,
            n_layers=3,
            margin=1.0,
            sampler="random",
            seed=42,
            device="cuda",
            lower_upper_bound=None,
            with_training=True,
        ):
            super().__init__(task, data_info, embed_size, lower_upper_bound)
    
            self.all_args = locals()
            self.loss_type = loss_type
            self.n_epochs = n_epochs
            self.lr = lr
            self.lr_decay = lr_decay
            self.epsilon = epsilon
            self.amsgrad = amsgrad
            self.reg = reg
            self.batch_size = batch_size
            self.num_neg = num_neg
            self.dropout_rate = dropout_rate
            self.n_layers = n_layers
            self.margin = margin
            self.sampler = sampler
            self.seed = seed
            self.device = device_config(device)
            self._check_params()
    
        def build_model(self):
            set_torch_seed(self.seed)
            self.torch_model = LightGCNModel(
                self.n_users,
                self.n_items,
                self.embed_size,
                self.n_layers,
                self.dropout_rate,
                self.user_consumed,
                self.device,
            )
    
        def _check_params(self):
            if self.task != "ranking":
                raise ValueError("LightGCN is only suitable for ranking")
            if self.loss_type not in ("cross_entropy", "focal", "bpr", "max_margin"):
                raise ValueError(f"unsupported `loss_type` for LightGCN: {self.loss_type}")
    
        @torch.inference_mode()
        def set_embeddings(self):
            self.torch_model.eval()
            embeddings = self.torch_model.embedding_propagation(use_dropout=False)
            self.user_embeds_np = embeddings[0].detach().cpu().numpy()
            self.item_embeds_np = embeddings[1].detach().cpu().numpy()
    ```
    

| Parameter | Default Value | Description |
| --- | --- | --- |
| task | (Required) | 추천 작업의 유형을 정의합니다. (예: 랭킹) |
| data_info | (Required) | 훈련과 추론에 필요한 정보를 포함하는 객체입니다. |
| loss_type | "bpr" | 모델 훈련에 사용되는 손실 함수 유형입니다. |
| embed_size | 16 | 임베딩 벡터의 크기를 나타냅니다. |
| n_epochs | 20 | 훈련의 에포크 수를 나타냅니다. |
| lr | 0.001 | 학습률입니다. |
| lr_decay | False | 학습률 감소를 사용할지 여부를 나타냅니다. |
| epsilon | 1e-8 | Adam 최적화에서 수치 안정성을 위해 분모에 추가되는 작은 상수입니다. |
| amsgrad | False | "On the Convergence of Adam and Beyond" 논문의 AMSGrad 변형을 사용할지의 여부입니다. |
| reg | None | 정규화 파라미터입니다. 양수 또는 None이어야 합니다. |
| batch_size | 256 | 훈련에 사용되는 배치 크기입니다. |
| num_neg | 1 | 각 양성 샘플에 대한 부정 샘플의 수를 나타냅니다. |
| dropout_rate | 0.0 | 노드가 드롭아웃 될 확률입니다. 0.0은 드롭아웃이 사용되지 않음을 의미합니다. |
| n_layers | 3 | GCN 계층의 수를 나타냅니다. |
| margin | 1.0 | max_margin 손실에서 사용되는 마진 값을 나타냅니다. |
| sampler | "random" | 부정 샘플링 전략입니다. ('random', 'unconsumed', 'popular' 중 하나) |
| seed | 42 | 무작위 시드 값입니다. |
| device | "cuda" | 연산을 수행할 장치를 나타냅니다. ("cpu" 또는 "cuda") |
| lower_upper_bound | None | rating 작업에 대한 점수의 하한과 상한을 나타냅니다. |
| with_training | True | 훈련과 관련된 추가 파라미터나 기능을 포함할지의 여부를 나타냅니다. (이 설명은 코드에서 직접적으로 제공되지 않았으나, 일반적인 의미로 해석되었습니다.) |

### **embed_size  32 고정 epochs 50~1000**  7개

### **epochs 50**

| loss | 0.6675550724289926 |
| --- | --- |
| roc_auc | 0.8723970818594675 |
| precision | 0.0988294314381271 |
| recall | 0.10143176407619994 |
| ndcg | 0.3083421681311364 |

### **epochs 100**

| loss | 0.7188938369022738 |
| --- | --- |
| roc_auc | 0.8815188697196754 |
| precision | 0.10953177257525083 |
| recall | 0.11722058788063154 |
| ndcg | 0.31903766168140474 |

### **epochs 1000**

| loss | 1.1907000049489085 |
| --- | --- |
| roc_auc | 0.8648312198965391 |
| precision | 0.11622073578595318 |
| recall | 0.13233993842996425 |
| ndcg | 0.36158267794917587 |

```python
['Silence of the Lambs, The (1991)']
['Forrest Gump (1994)']
['Braveheart (1995)']
['Jurassic Park (1993)']
['Star Wars: Episode IV - A New Hope (1977)']
["Schindler's List (1993)"]
['Toy Story (1995)']
==========================임베딩 사이즈 : 32, 에포크 : 50 =============================
['Silence of the Lambs, The (1991)']
['Forrest Gump (1994)']
['Jurassic Park (1993)']
['Star Wars: Episode IV - A New Hope (1977)']
['Braveheart (1995)']
['Fugitive, The (1993)']
['Toy Story (1995)']
==========================임베딩 사이즈 : 32, 에포크 : 100 =============================
==========================임베딩 사이즈 : 32, 에포크 : 1000 =============================
```

### **embed_size  64 고정 epochs 50~1000 5~7개**

### **epochs 50**

| loss | 0.6888450731817461 |
| --- | --- |
| roc_auc | 0.8780701722927594 |
| precision | 0.10518394648829432 |
| recall | 0.11060219016698679 |
| ndcg | 0.31771938560195023 |

### **epochs 100**

| loss | 0.7556582918426417 |
| --- | --- |
| roc_auc | 0.8822069434701935 |
| precision | 0.11404682274247492 |
| recall | 0.12157058102596818 |
| ndcg | 0.33249304238838445 |

### **epochs 1000**

| loss | 1.196622766709713 |
| --- | --- |
| roc_auc | 0.8636465730261751 |
| precision | 0.12157190635451504 |
| recall | 0.14316534898629957 |
| ndcg | 0.38070191870965436 |

```python
['Silence of the Lambs, The (1991)']
['Forrest Gump (1994)']
['Jurassic Park (1993)']
['Braveheart (1995)']
['Star Wars: Episode IV - A New Hope (1977)']
['Toy Story (1995)']
['Fugitive, The (1993)']
==========================임베딩 사이즈 : 64, 에포크 : 50 =============================
['Star Wars: Episode IV - A New Hope (1977)']
['Jurassic Park (1993)']
['Silence of the Lambs, The (1991)']
['Forrest Gump (1994)']
['Toy Story (1995)']
==========================임베딩 사이즈 : 64, 에포크 : 100 =============================
==========================임베딩 사이즈 : 64, 에포크 : 1000 =============================
```

### **embed_size  128 고정 epochs 50~1000** 1~6개

### **epochs 50**

| loss | 0.716364958545461 |
| --- | --- |
| roc_auc | 0.8821943701694291 |
| precision | 0.112876254180602 |
| recall | 0.12261600697189515 |
| ndcg | 0.3254422588748364 |

### **epochs 100**

| loss | 0.800694605040523 |
| --- | --- |
| roc_auc | 0.8816639524430555 |
| precision | 0.11822742474916391 |
| recall | 0.1286562083998558 |
| ndcg | 0.3486267351008592 |

### **epochs 1000**

| loss | 1.1432882939184 |
| --- | --- |
| roc_auc | 0.863262030210714 |
| precision | 0.12491638795986625 |
| recall | 0.1476914264039193 |
| ndcg | 0.3890408169009991 |

```python
['Silence of the Lambs, The (1991)']
['Forrest Gump (1994)']
['Star Wars: Episode IV - A New Hope (1977)']
['Jurassic Park (1993)']
['Braveheart (1995)']
['Toy Story (1995)']
==========================임베딩 사이즈 : 128, 에포크 : 50 =============================
['Star Wars: Episode IV - A New Hope (1977)']
==========================임베딩 사이즈 : 128, 에포크 : 100 =============================
==========================임베딩 사이즈 : 128, 에포크 : 1000 =============================
```

