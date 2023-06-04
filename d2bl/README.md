## Info

+ In the folder `hebbian` there are the experiments for finetuning [vgg16](https://github.com/r1cc4r2o/D2BL/blob/main/d2bl/hebbian/vgg19_hebb.ipynb) or [rn50](https://github.com/r1cc4r2o/D2BL/blob/main/d2bl/hebbian/resnet_hebb.ipynb) with one or [more](https://github.com/r1cc4r2o/D2BL/blob/main/d2bl/hebbian/resnet_deep_hebb.ipynb) Hebbian layer.

+ In the folder `pseudo-attention` there is our [LinW_module](https://github.com/r1cc4r2o/D2BL/blob/main/d2bl/pseudo-attention/0_pseudo-attention.ipynb) trainable from scratch on cifar10 or mnist. Additionally, there is the implementation of the [LinW_module](https://github.com/r1cc4r2o/D2BL/blob/main/d2bl/pseudo-attention/1_pseudo-attention-mut-cross.ipynb) which introduce crossing-over and mutations on the saliency map. Furthermore, there are experiments for finetuning it using [vgg19](https://github.com/r1cc4r2o/D2BL/blob/main/d2bl/pseudo-attention/2_vgg19_linW.ipynb) or [rn50](https://github.com/r1cc4r2o/D2BL/blob/main/d2bl/pseudo-attention/2_rn50_linW.ipynb) on cifar10.

+ In the folder `kde_learning` there are the experiments that we have done using KDE on past activations to get novel representations and condition the network on them. Specifically, in the notebook [0_kde_learning.ipynb](https://github.com/r1cc4r2o/D2BL/blob/main/d2bl/kde_learning/0_kde_learning.ipynb) you can find the implementation of the network which stack together the past activations in a single vector. Whereas, in the second notebook [1_kde_learning.ipynb](https://github.com/r1cc4r2o/D2BL/blob/main/d2bl/kde_learning/1_kde_learning.ipynb) you can find the implementation of the network which consider each learnt past representation as a point in the space. Further details can be found in the report [here](404).

+ The folder simulations contain the experiments using our layer for making a car drive on its own ([1_self_driving.ipynb](https://github.com/r1cc4r2o/D2BL/blob/main/d2bl/simulations/1_self_driving.ipynb)) and to simulate the flocking behaviour of boids ([0_flocking_behavior.ipynb](https://github.com/r1cc4r2o/D2BL/blob/main/d2bl/simulations/0_flocking_behavior.ipynb)).





