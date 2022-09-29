# Deep iForest

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
