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

## Takeaways
DIF provides easy APIs in a sklearn style, that is, we first instantiate the model class by giving the parameters  
then, the instantiated model can be used to fit and predict data

```python
from algorithms.dif import DIF
model_configs = {'n_ensemble':50, 'n_estimators':6}
model = DIF(**model_configs)
model.fit(X_train)
score = model.predict(X_test)
```


---
## Datasets

We retain two small tabular datasets (*Pageblocks* and *Shuttle*) and one graph dataset (*MMP*) as examples.

*Pageblocks* and *Shuttle* are obtained from https://www.dbs.ifi.lmu.de/research/outlier-evaluation/DAMI/  
*MMP* is obtained from https://chrsmrrs.github.io/datasets/docs/datasets/


  
  
Copyright of dataset *MSL*:
```
Copyright Assertion

Copyright (c) 2018, California Institute of Technology ("Caltech").  U.S. Government sponsorship acknowledged.

All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

•	Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
•	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
•	Neither the name of Caltech nor its operating division, the Jet Propulsion Laboratory, nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
