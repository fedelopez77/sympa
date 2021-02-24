# sympa
Graph embeddings in SYMetric sPAces

## Requirements
 - Python == 3.7 
 - Pytorch == 1.5.1: ```conda install pytorch==1.5.1 torchvision==0.6.1 [cpuonly | cudatoolkit=10.2] -c pytorch```
 - [Geoopt](https://github.com/geoopt/geoopt) >= 0.3.1: install from repository is advised: ```pip install git+https://github.com/geoopt/geoopt.git```
 - [XiTorch](https://github.com/xitorch/xitorch): for working with the compact dual only
 - networkx and networkit: for preprocessing only
 - matplotlib: for preprocessing only
 - tensorboardx
 - tqdm
 
## Considerations
The method `inner` is implemented for both the Upper Half space and the Bounded domain model.
With this, experiments can be run with `RiemannianAdam`.
However, we found them to be very unstable, therefore all experiments reported in the paper were run with `RiemannianSGD`
