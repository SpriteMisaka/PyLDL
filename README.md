# PyLDL

LDL(Label Distribution Learning) toolkit implemented in python, including:

+ 6 LDL algorithms proposed by [Geng](https://github.com/SpriteMisaka/PyLDL/blob/main/LDL.pdf): SA_BFGS, SA_IIS, AA_KNN, AA_BP, PT_Bayes, PT_SVM;
+ 10 LDL metrics: chebyshev, clark, canberra, kl_divergence, cosine, intersection, euclidean, sorensen, squared_chi2, fidelity;
+ 15 LDL datasets: Human_Gene, Movie, Natural_Scene, SBU_3DFE, SJAFFE, Yeast_alpha, Yeast_cdc, Yeast_cold, Yeast_diau, Yeast_dtt, Yeast_elu, Yeast_heat, Yeast_spo, Yeast_spo5, Yeast_spoem.

Some codes are from [https://github.com/wangjing4research/LDL](https://github.com/wangjing4research/LDL).

SA_IIS is extremely slow because of my poor programming. Please help me if you have any better idea! :)

## Usage

```python
from utils import load_dataset
from algorithms import SA_BFGS

dataset_name = 'SJAFFE'
X, y = load_dataset(dataset_name)

model = SA_BFGS()
model.fit(X, y)
```

## Experiment

For each algorithm, a ten-fold cross validation is performed, repeated 10 times with SJAFFE dataset and the average metrics are recorded.

| Algorithm | Chebyshev(↓) |  Clark(↓)  | Canberra(↓) |   K-L(↓)   |  Cosine(↑) | Intersection(↑) |
|:---------:|:------------:|:----------:|:-----------:|:----------:|:----------:|:---------------:|
|  SA-BFGS  |  **0.0926**  |   0.3605   |    0.7359   | **0.0506** | **0.9539** |    **0.8777**   |
|   SA-IIS  |    0.1004    |   0.3606   |    0.7463   | **0.0506** |   0.9520   |      0.8733     |
|   AA-KNN  |    0.0978    | **0.3483** |  **0.7164** |   0.0528   |   0.9497   |      0.8766     |
|   AA-BP   |    0.1192    |   0.4288   |    0.8916   |   0.0740   |   0.9302   |      0.8475     |
|  PT-Bayes |    0.1278    |   0.4539   |    0.9149   |   0.0841   |   0.9214   |      0.8425     |
|   PT-SVM  |    0.1199    |   0.4239   |    0.8728   |   0.0720   |   0.9318   |      0.8503     |
