### Too many Features

- Curse of Dimensionality

`Feature가 증가하면 그 만큼 더 많은 sample들이 필요하다.`

      5개의 Sample에 대한 특성으로 각 sample들 간의 거리를 이용한다고 하자.
      만일 Feature의 개수가 늘어나 각 sample을 정의할 수 있는 차원이 늘어난다고 하자.

      이제 차원이 늘어남에 따라 각 sample 간의 거리는 점점 멀어질 것이고, 결국은 각 feature 간의 거리가 특징이 없어질 정도로 비슷하게 멀어진다.
      따라서 차원이 늘어남에 따라 Model 학습에 필요하게 되는 Sample의 수는 기하급수적으로 늘어난다.

<br>

- Solution 

      1. Feature Selection
      2. Feature Extraction
        - Linear Method : PCA
        - Non Linear Method : t-SNE



    