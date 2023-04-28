# Deep iForest

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/deep-isolation-forest-for-anomaly-detection/anomaly-detection-on-kaggle-credit-card-fraud)](https://paperswithcode.com/sota/anomaly-detection-on-kaggle-credit-card-fraud?p=deep-isolation-forest-for-anomaly-detection)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/deep-isolation-forest-for-anomaly-detection/anomaly-detection-on-forest-covertype)](https://paperswithcode.com/sota/anomaly-detection-on-forest-covertype?p=deep-isolation-forest-for-anomaly-detection)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/deep-isolation-forest-for-anomaly-detection/anomaly-detection-on-nb15-backdoor)](https://paperswithcode.com/sota/anomaly-detection-on-nb15-backdoor?p=deep-isolation-forest-for-anomaly-detection)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/deep-isolation-forest-for-anomaly-detection/anomaly-detection-on-nb15-dos)](https://paperswithcode.com/sota/anomaly-detection-on-nb15-dos?p=deep-isolation-forest-for-anomaly-detection)


This repository is the source code of the paper "**Deep Isolation Forest for Anomaly Detection**" published in TKDE (April 2023).  (see full paper at https://arxiv.org/abs/2206.06602 or https://ieeexplore.ieee.org/document/10108034/ )   


### How to use?


DIF provides easy APIs like the sklearn style.
We first instantiate the model class by giving the parameters  
then, the instantiated model can be used to fit and predict data

```python
from algorithms.dif import DIF
model_configs = {'n_ensemble':50, 'n_estimators':6}
model = DIF(**model_configs)
model.fit(X_train)
score = model.predict(X_test)
```

:boom:**Note:** 
- DIF is also included in our `DeepOD` python library. Please see https://github.com/xuhongzuo/DeepOD 
- Please also see the Zhihu blog (in Chinese) https://zhuanlan.zhihu.com/p/625557221 


### Citation

Please consider citing our paper if you find this repository useful.  

H. Xu, G. Pang, Y. Wang and Y. Wang, "Deep Isolation Forest for Anomaly Detection," in IEEE Transactions on Knowledge and Data Engineering, doi: 10.1109/TKDE.2023.3270293.

```
@ARTICLE{xu2023deep,
  author={Xu, Hongzuo and Pang, Guansong and Wang, Yijie and Wang, Yongjun},
  journal={IEEE Transactions on Knowledge and Data Engineering}, 
  title={Deep Isolation Forest for Anomaly Detection}, 
  year={2023},
  volume={},
  number={},
  pages={1-14},
  doi={10.1109/TKDE.2023.3270293}}

```


---
