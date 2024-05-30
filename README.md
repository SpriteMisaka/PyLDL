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
  + ([González et al. 2021a](https://github.com/SpriteMisaka/PyLDL/blob/main/bibliography/gonz%C3%A1lez2021a.pdf))[*Inf. Sci.*]: `SSG_LDL`$^4$.
  + ([González et al. 2021b](https://github.com/SpriteMisaka/PyLDL/blob/main/bibliography/gonz%C3%A1lez2021b.pdf))[*Inf. Fusion*]: `DF_LDL`.
  + ([Wang and Geng 2021a](https://github.com/SpriteMisaka/PyLDL/blob/main/bibliography/wang2021a.pdf))[*IJCAI*]: `LDL_HR`$^3$.
  + ([Wang and Geng 2021b](https://github.com/SpriteMisaka/PyLDL/blob/main/bibliography/wang2021b.pdf))[*ICML*]: `LDLM`$^3$.
  + ([Jia et al. 2021](https://github.com/SpriteMisaka/PyLDL/blob/main/bibliography/jia2021.pdf))[*TKDE*]: `LDL_SCL`.
  + ([Jia et al. 2023a](https://github.com/SpriteMisaka/PyLDL/blob/main/bibliography/jia2023a.pdf))[*TKDE*]: `LDL_LRR`.
  + ([Jia et al. 2023b](https://github.com/SpriteMisaka/PyLDL/blob/main/bibliography/jia2023b.pdf))[*TNNLS*]: `LDL_DPA`.
  + ([Wen et al. 2023](https://github.com/SpriteMisaka/PyLDL/blob/main/bibliography/wen2023.pdf))[*ICCV*]: `CAD`$^1$, `QFD2`$^1$, and `CJS`$^1$.
+ LE algorithms:
  + ([Xu, Liu, and Geng 2019](https://github.com/SpriteMisaka/PyLDL/blob/main/bibliography/xu2019.pdf))[*TKDE*]: `FCM`, `KM`, `LP`, `ML`, and `GLLE`.
  + ([Xu et al. 2020](https://github.com/SpriteMisaka/PyLDL/blob/main/bibliography/xu2020.pdf))[*ICML*]: `LEVI`.
  + ([Zheng, Zhu, and Tang 2023](https://github.com/SpriteMisaka/PyLDL/blob/main/bibliography/zheng2023.pdf))[*CVPR*]: `LIBLE`.
+ LDL metrics: `chebyshev`, `clark`, `canberra`, `kl_divergence`, `cosine`, `intersection`, etc.
+ Structured LDL datasets: *Human_Gene*, *Movie*, *Natural_Scene*, *s-BU_3DFE*, *s-JAFFE*, *Yeast*, etc.
+ LDL applications:
  + Facial emotion recognition (supported datasets: [*JAFFE*](https://zenodo.org/records/3451524)).
  + ([Shirani et al. 2019](https://github.com/SpriteMisaka/PyLDL/blob/main/bibliography/shirani2019.pdf))[*ACL*]: Emphasis selection (supported datasets: [*SemEval2020*](https://github.com/RiTUAL-UH/SemEval2020_Task10_Emphasis_Selection); pre-trained GloVe embeddings can be downloaded [here](https://nlp.stanford.edu/projects/glove/)).
  + ([Wu et al. 2019](https://github.com/SpriteMisaka/PyLDL/blob/main/bibliography/wu2019.pdf))[*ICCV*]: Lesion counting (supported datasets: [*ACNE04*](https://drive.google.com/drive/folders/18yJcHXhzOv7H89t-Lda6phheAicLqMuZ)).

> $^1$ Technically, these methods are only suitable for totally ordered labels.
>
> $^2$ These are algorithms for incomplete LDL, so you should use `pyldl.utils.random_missing` to generate the missing label distribution matrix and the corresponding mask matrix in the experiments.
>
> $^3$ These are LDL classifiers, so you should use `predict_proba` to get label distributions and `predict` to get predicted labels.
>
> $^4$ These are oversampling algorithms for LDL, therefore you should use `fit_transform` to generate synthetic samples.

## Installation

PyLDL is now available on [PyPI](https://pypi.org/project/python-ldl/). Use the following command to install.

```shell
pip install python-ldl
```

To install the newest version, you can clone this repo and run the `setup.py` file.

```shell
python setup.py install
```

## Usage

Here is an example of using PyLDL.

```python
from pyldl.utils import load_dataset
from pyldl.algorithms import SA_BFGS
from pyldl.metrics import score

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
3. Download LDL Package [here](http://palm.seu.edu.cn/xgeng/LDL/download.htm).
3. Get the package directory of PyLDL (...\\Lib\\site-packages\\pyldl).
4. Place the *LDLPackage_v1.2* folder into the *matlab_algorithms* folder.

Now, you can load the original implementation of the method, e.g.:

```python
from pyldl.matlab_algorithms import SA_IIS
```

You can visualize the performance of any model on the artificial dataset ([Geng 2016](https://github.com/SpriteMisaka/PyLDL/blob/main/bibliography/geng2016.pdf)) with the `pyldl.utils.plot_artificial` function, e.g.:

```python
from pyldl.algorithms import LDSVR, SA_BFGS, SA_IIS, AA_KNN, PT_Bayes, GLLE, LIBLE
from pyldl.utils import plot_artificial

methods = ['LDSVR', 'SA_BFGS', 'SA_IIS', 'AA_KNN', 'PT_Bayes', 'GLLE', 'LIBLE']

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

| <img src="https://github.com/SpriteMisaka/PyLDL/blob/main/visualization/GLLE.jpg?raw=true" width=300> | <img src="https://github.com/SpriteMisaka/PyLDL/blob/main/visualization/LIBLE.jpg?raw=true" width=300> |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
|                            `GLLE`                            |                           `LIBLE`                            |

Enjoy! :)

## Experiments

For each algorithm, a ten-fold cross validation is performed, repeated 10 times with *s-JAFFE* dataset and the average metrics are recorded. Therefore, the results do not fully describe the performance of the model.

Results of ours are as follows.

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
matplotlib>=3.6.1
numpy>=1.22.3
qpsolvers>=4.0.0
quadprog>=0.1.11
scikit-fuzzy>=0.4.2
scikit-learn>=1.0.2
scipy>=1.8.0
tensorflow>=2.8.0
tensorflow-addons>=0.22.0
tensorflow-probability>=0.16.0
```

