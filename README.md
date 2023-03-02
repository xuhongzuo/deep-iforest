# Deep iForest

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/deep-isolation-forest-for-anomaly-detection/anomaly-detection-on-kaggle-credit-card-fraud)](https://paperswithcode.com/sota/anomaly-detection-on-kaggle-credit-card-fraud?p=deep-isolation-forest-for-anomaly-detection)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/deep-isolation-forest-for-anomaly-detection/anomaly-detection-on-forest-covertype)](https://paperswithcode.com/sota/anomaly-detection-on-forest-covertype?p=deep-isolation-forest-for-anomaly-detection)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/deep-isolation-forest-for-anomaly-detection/anomaly-detection-on-nb15-backdoor)](https://paperswithcode.com/sota/anomaly-detection-on-nb15-backdoor?p=deep-isolation-forest-for-anomaly-detection)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/deep-isolation-forest-for-anomaly-detection/anomaly-detection-on-nb15-dos)](https://paperswithcode.com/sota/anomaly-detection-on-nb15-dos?p=deep-isolation-forest-for-anomaly-detection)


This repository is the source code of the paper "Deep Isolation Forest for Anomaly Detection" (see full paper at https://arxiv.org/abs/2206.06602 )   
Please consider citing our paper if you find this repository useful.  

```
@article{xu2022deep,
  title={Deep Isolation Forest for Anomaly Detection},
  author={Xu, Hongzuo and Pang, Guansong and Wang, Yijie and Wang, Yongjun},
  journal={arXiv preprint arXiv:2206.06602},
  year={2022}
}
```

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

---
