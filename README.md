# Deep iForest

This repository is the source code the paper: https://arxiv.org/abs/2206.06602  
Please cite our paper if you use this repository.  

```
@article{xu2022deep,
  title={Deep Isolation Forest for Anomaly Detection},
  author={Xu, Hongzuo and Pang, Guansong and Wang, Yijie and Wang, Yongjun},
  journal={arXiv preprint arXiv:2206.06602},
  year={2022}
}
```


## Reproduction of experiment results
All the experiment results reported in our paper can be well reproduced. 

#### experiments on tabular data (Sec. 4.2.1)
use `python main.py --runs 10 --model dif` to run our model DIF,  

#### experiments on graph and time-series data (Sec. 4.2.2)
use `python main_graph.py --runs 10 --model dif` to perform the experiments on graph data  
GLocalKD and InfoGraph can be directly used after downloading their implementation from Github. 

use `python main_ts.py --runs 10 --model dif` to perform experiments on time-series data
GDN and Omni are also publicly aviable and can be directly used after downloading from Github

  
#### Ablation study (Sec. 4.3)
Five ablated varients DIF-AE, DIF-DSVDD, RR-COPOD, RR-KNN, and RR-LOF are supported in this project as well. 


#### Robustness w.r.t. Contamination Ratios (Sec. 4.4.1)
Please add `--contamination 0.1` when performing `main.py`

#### Scalability Test (Sec. 4.4.2)
The synthetic datasets can be created by `create_scal_data.py`.  
we record the execution time in the final record files. After each running, a record file will be generated. 

#### Convergence (Sec. 4.4.3)
Change the `--n_ensemble 50` (number of representations) and `--n_estimators 6` (number of trees per representation) to other settings.  


#### Sensitivity Test (Sec. 4.4.4)
Different network structures are implemented as well, please change the argument `--network_name`


---
## Datasets

We retain two small tabular datasets (*Ad* and *Cardio*), one time-series dataset (*MSL*), and one graph dataset (*MMP*) as examples.
For other datasets, we provide links in the appendix of our paper.

*Ad* and *Cardio* are obtained from https://archive.ics.uci.edu/  
*MSL* is obtinaed from https://github.com/khundman/telemanom  
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
