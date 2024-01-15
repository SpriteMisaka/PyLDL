# PyLDL

Label distribution learning (LDL) and label enhancement (LE) toolkit implemented in python, including:

+ LDL algorithms:
  + ([Geng, Yin, and Zhou 2013](https://github.com/SpriteMisaka/PyLDL/blob/main/bibliography/geng2013.pdf))[*TPAMI*]: `CPNN`$^1$.
  + ([Geng and Hou 2015](https://github.com/SpriteMisaka/PyLDL/blob/main/bibliography/geng2015.pdf))[*IJCAI*]: `LDSVR`.
  + ⭐([Geng 2016](https://github.com/SpriteMisaka/PyLDL/blob/main/bibliography/geng2016.pdf))[*TKDE*]: `SA_BFGS`, `SA_IIS`, `AA_KNN`, `AA_BP`, `PT_Bayes`, and `PT_SVM`.
  + ([Yang, Sun, and Sun 2017](https://github.com/SpriteMisaka/PyLDL/blob/main/bibliography/yang2017.pdf))[*AAAI*]: `BCPNN` and `ACPNN`.
  + ([Xu and Zhou 2017](https://github.com/SpriteMisaka/PyLDL/blob/main/bibliography/xu2017.pdf))[*IJCAI*]: `IncomLDL`$^2$.
  + ([Shen et al. 2017](https://github.com/SpriteMisaka/PyLDL/blob/main/bibliography/shen2017.pdf))[*NeurIPS*]: `LDLF`.
  + ([Wang and Geng 2019](https://github.com/SpriteMisaka/PyLDL/blob/main/bibliography/wang2019.pdf))[*IJCAI*]: `LDL4C`$^3$.
  + ([Shen et al. 2020](https://github.com/SpriteMisaka/PyLDL/blob/main/bibliography/shen2020.pdf))[*南京理工大学学报* (Chinese)]: `AdaBoostLDL`.
  + ([González et al. 2021a](https://github.com/SpriteMisaka/PyLDL/blob/main/bibliography/gonz%C3%A1lez2021a.pdf))[*Information Sciences*]: `SSG_LDL`$^4$.
  + ([González et al. 2021b](https://github.com/SpriteMisaka/PyLDL/blob/main/bibliography/gonz%C3%A1lez2021b.pdf))[*Information Fusion*]: `DF_LDL`.
  + ([Wang and Geng 2021a](https://github.com/SpriteMisaka/PyLDL/blob/main/bibliography/wang2021a.pdf))[*IJCAI*]: `LDL_HR`$^3$.
  + ([Wang and Geng 2021b](https://github.com/SpriteMisaka/PyLDL/blob/main/bibliography/wang2021b.pdf))[*ICML*]: `LDLM`$^3$.
  + ([Jia et al. 2021](https://github.com/SpriteMisaka/PyLDL/blob/main/bibliography/jia2021.pdf))[*TKDE*]: `LDL_SCL`.
  + ([Jia et al. 2023](https://github.com/SpriteMisaka/PyLDL/blob/main/bibliography/jia2023.pdf))[*TKDE*]: `LDL_LRR`.
  + ([Wen et al. 2023](https://github.com/SpriteMisaka/PyLDL/blob/main/bibliography/wen2023.pdf))[*ICCV*]: `CAD`$^1$, `QFD2`$^1$, and `CJS`$^1$.
+ LE algorithms:
  + ([Xu, Liu, and Geng 2019](https://github.com/SpriteMisaka/PyLDL/blob/main/bibliography/xu2019.pdf))[*TKDE*]: `FCM`, `KM`, `LP`, `ML` and `GLLE`.
  + ([Xu et al. 2020](https://github.com/SpriteMisaka/PyLDL/blob/main/bibliography/xu2020.pdf))[*ICML*]: `LEVI`.

+ LDL metrics: `chebyshev`, `clark`, `canberra`, `kl_divergence`, `cosine`, `intersection`, etc.
+ LDL datasets: *Human_Gene*, *Movie*, *Natural_Scene*, *s-BU_3DFE*, *s-JAFFE*, *Yeast*, etc.

> $^1$ Technically, these methods are only suitable for totally ordered labels.
>
> $^2$ These are algorithms for incomplete LDL, so you should use `utils.random_missing` to generate the missing label distribution matrix and the corresponding mask matrix in the experiments.
>
> $^3$ These are LDL classifiers, so you should use `predict_proba` to get label distributions and `predict` to get predicted labels.
>
> $^4$ These are oversampling algorithms for LDL, therefore you should use `fit_transform` to generate synthetic samples.

## Usage

Here is an example of using PyLDL.

```python
from utils import load_dataset
from algorithms import SA_BFGS
from metrics import score

from sklearn.model_selection import train_test_split

dataset_name = 'SJAFFE'
X, y = load_dataset(dataset_name)
X_train, X_test, y_train, y_test = train_test_split(X, y)

model = SA_BFGS()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(score(y_test, y_pred))
```

For those who would like to use the original implementation:

1. Install MATLAB.
2. Install MATLAB engine for python.
3. Download LDL Package from [here](http://palm.seu.edu.cn/xgeng/LDL/download.htm).
4. Place the *LDLPackage_v1.2* folder into the *matlab_algorithms* folder.

Now, you can load the original implementation of the method, e.g.:

```python
from matlab_algorithms import SA_IIS
```

You can visualize the performance of any model on the artificial dataset ([Geng 2016](https://github.com/SpriteMisaka/PyLDL/blob/main/bibliography/geng2016.pdf)) with the `utils.plot_artificial` function, e.g.:

```python
from algorithms import LDSVR, SA_BFGS, SA_IIS, AA_KNN, PT_Bayes
from utils import plot_artificial

methods = ['LDSVR', 'SA_BFGS', 'SA_IIS', 'AA_KNN', 'PT_Bayes']

plot_artificial(model=None, figname='GT')
for i in methods:
    plot_artificial(model=eval(f'{i}()'), figname=i)
```

The output images are as follows.

| <img src="https://github.com/SpriteMisaka/PyLDL/blob/main/visualization/GT.jpg?raw=true" width=300> | <img src="https://github.com/SpriteMisaka/PyLDL/blob/main/visualization/LDSVR.jpg?raw=true" width=300> |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
|                        (Ground Truth)                        |                           `LDSVR`                            |

| <img src="https://github.com/SpriteMisaka/PyLDL/blob/main/visualization/SA_BFGS.jpg?raw=true" width=300> | <img src="https://github.com/SpriteMisaka/PyLDL/blob/main/visualization/SA_IIS.jpg?raw=true" width=300> |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
|                          `SA_BFGS`                           |                           `SA_IIS`                           |

| <img src="https://github.com/SpriteMisaka/PyLDL/blob/main/visualization/AA_KNN.jpg?raw=true" width=300> | <img src="https://github.com/SpriteMisaka/PyLDL/blob/main/visualization/PT_Bayes.jpg?raw=true" width=300> |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
|                           `AA_KNN`                           |                          `PT_Bayes`                          |

Enjoy! :)

## Experiments

For each algorithm, a ten-fold cross validation is performed, repeated 10 times with *s-JAFFE* dataset and the average metrics are recorded. Therefore, the results do not fully describe the performance of the model.

Results of ours are as follows.

(Novel LDL Algorithms)

| Algorithm |    Cheby.(↓)    |    Clark(↓)     |     Can.(↓)     |     K-L(↓)      |     Cos.(↑)     |     Int.(↑)     |
| :-------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
|  LDL-LRR  | **.085 ± .010** | **.323 ± .027** | **.656 ± .055** | **.041 ± .008** | **.962 ± .008** | **.891 ± .010** |
|  LDL-SCL  |   .087 ± .008   |   .336 ± .024   |   .688 ± .051   |   .044 ± .006   |   .959 ± .006   |   .885 ± .009   |
|   LDLF    |   .092 ± .011   |   .353 ± .035   |   .721 ± .076   |   .049 ± .010   |   .954 ± .009   |   .879 ± .013   |

(LDL Algorithms using `SA_BFGS` as the Base Estimator)

| Algorithm  |    Cheby.(↓)    |    Clark(↓)     |     Can.(↓)     |     K-L(↓)      |     Cos.(↑)     |     Int.(↑)     |
| :--------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
|  DF-BFGS   | **.083 ± .009** | **.306 ± .025** | **.624 ± .052** | **.039 ± .007** | **.963 ± .007** | **.894 ± .009** |
|  SSG-BFGS  |   .090 ± .090   |   .343 ± .343   |   .699 ± .699   |   .047 ± .047   |   .957 ± .957   |   .883 ± .883   |
| BFGS-AdaB. |   .087 ± .009   |   .327 ± .025   |   .666 ± .050   |   .042 ± .007   |   .961 ± .007   |   .888 ± .009   |
| (Baseline) |   .092 ± .010   |   .361 ± .029   |   .735 ± .060   |   .051 ± .009   |   .954 ± .009   |   .878 ± .011   |

(Incomplete LDL)

|       Algorithm        |    Cheby.(↓)    |    Clark(↓)     |     Can.(↓)     |     Cos.(↑)     |     Int.(↑)     |
| :--------------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
| SA-BFGS (10% Missing)  |   .101 ± .010   |   .396 ± .031   |   .815 ± .065   |   .943 ± .010   |   .863 ± .011   |
| IncomLDL (10% Missing) | **.095 ± .009** | **.354 ± .022** | **.735 ± .046** | **.956 ± .006** | **.876 ± .008** |

|       Algorithm        |    Cheby.(↓)    |    Clark(↓)     |     Can.(↓)     |     Cos.(↑)     |     Int.(↑)     |
| :--------------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
| SA-BFGS (40% Missing)  |   .108 ± .012   |   .404 ± .029   |   .831 ± .063   |   .938 ± .011   |   .858 ± .012   |
| IncomLDL (40% Missing) | **.104 ± .010** | **.371 ± .021** | **.773 ± .046** | **.950 ± .006** | **.869 ± .009** |

(Classical LDL Algorithms)

| Algorithm |    Cheby.(↓)    |    Clark(↓)     |     Can.(↓)     |     K-L(↓)      |     Cos.(↑)     |     Int.(↑)     |
| :-------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
|  SA-BFGS  | **.092 ± .010** |   .361 ± .029   |   .735 ± .060   | **.051 ± .009** | **.954 ± .009** | **.878 ± .011** |
|  SA-IIS   |   .100 ± .009   |   .361 ± .023   |   .746 ± .050   | **.051 ± .008** |   .952 ± .007   |   .873 ± .009   |
|  AA-kNN   |   .098 ± .011   | **.349 ± .029** | **.716 ± .062** |   .053 ± .010   |   .950 ± .009   |   .877 ± .011   |
|   AA-BP   |   .120 ± .012   |   .426 ± .025   |   .889 ± .057   |   .073 ± .010   |   .931 ± .010   |   .848 ± .011   |
| PT-Bayes  |   .116 ± .011   |   .425 ± .031   |   .874 ± .064   |   .073 ± .012   |   .932 ± .011   |   .850 ± .012   |
|  PT-SVM   |   .117 ± .012   |   .422 ± .027   |   .875 ± .057   |   .072 ± .011   |   .932 ± .011   |   .850 ± .011   |

Results of the original MATLAB implementation ([Geng 2016](https://github.com/SpriteMisaka/PyLDL/blob/main/bibliography/geng2016.pdf)) are as follows.

| Algorithm |    Cheby.(↓)    |    Clark(↓)     |     Can.(↓)     |     K-L(↓)      |     Cos.(↑)     |     Int.(↑)     |
| :-------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
|  SA-BFGS  | **.107 ± .015** | **.399 ± .044** | **.820 ± .103** | **.064 ± .016** | **.940 ± .015** | **.860 ± .019** |
|  SA-IIS   |   .117 ± .015   |   .419 ± .034   |   .875 ± .086   |   .070 ± .012   |   .934 ± .012   |   .851 ± .016   |
|  AA-kNN   |   .114 ± .017   |   .410 ± .050   |   .843 ± .113   |   .071 ± .023   |   .934 ± .018   |   .855 ± .021   |
|   AA-BP   |   .130 ± .017   |   .510 ± .054   |   1.05 ± .124   |   .113 ± .030   |   .908 ± .019   |   .824 ± .022   |
| PT-Bayes  |   .121 ± .016   |   .430 ± .035   |   .904 ± .086   |   .074 ± .014   |   .930 ± .016   |   .846 ± .016   |
|  PT-SVM   |   .127 ± .017   |   .457 ± .039   |   .935 ± .074   |   .086 ± .016   |   .920 ± .014   |   .839 ± .015   |

## Requirements

```
keras==2.8.0
matplotlib==3.6.1
numpy==1.22.3
qpsolvers==4.0.0
quadprog==0.1.11
scikit-fuzzy==0.4.2
scikit-learn==1.0.2
scipy==1.8.0
tensorflow==2.8.0
tensorflow-addons==0.22.0
tensorflow-probability==0.16.0
```

