# D2BL: Dynamic Distribution-driven Backpropagation Learning
> [**D2BL: Investigation on distribution-driven backpropagation learning**](https://drive.google.com/file/d/1dBjPFGAVfrP30t2qLfGGunRPqu2ByOj-/view?usp=sharing),            
> [Andrea Coppari](https://it.linkedin.com/in/andreacoppari1005), [Riccardo Tedoldi](https://www.instagram.com/riccardotedoldi/)
Supervisor: [Andrea Ferigo](https://it.linkedin.com/in/andrea-ferigo), [Giovanni Iacca](https://it.linkedin.com/in/giovanniiacca)  
> *Project Bio-inspired, Spring 2023* 
<p align="center">D2BL with KDE learning</p>
<p align="center">
 <img src="img/D2BL-2.svg" width="45%">
 <img src="img/D2BL.svg" width="45%" >
</p>
<p align="center">D2BL with pseudo-attention mechanism with and without crossing-over and mutations</p>
<p align="center">
 <img src="img/pseudo-attention.svg" width="45%" >
 <img src="img/pseudo-attention-crossingovermutations.svg" width="45%" >
</p>
<p align="center">Pseudo-attention mechanism for fine-tuning</p>
<p align="center">
 <img src="img/pseudo-attention-crossingovermutations-finetuning.svg" width="66%" >
</p>


## Overview

In this repo, we report novel experiments to get activations conditioned on the representations extracted from previous stages of the network. We are trying to simulate what already happens in the brain. 

The brain initially processes stimuli at a low level, after which the signal is sent to higher cognitive areas involved in mental processes such as attention, memory, language, problem-solving, decision making and creativity. These higher cognitive areas are distributed throughout the brain, including the prefrontal cortex, hippocampus, frontal lobe and others, and various combinations of these areas work together to support different cognitive tasks. 

Specifically, our layer tries to exploit novel representations using this idea. The first idea that we propose combine the past representations using KDE (kernel density estimation) in two different ways. Whereas, with the second idea we combine the past activations using a pseudo-attention mechanism from which we get a saliency map that is used as a weighting factor of the activations of the immediately preceding layer. Once we are computing the gradients during backprop, the operations introduced so far will influence the gradients in extracting novel representations.
## Features

The structure of the repository is as follows:

+ The `d2bl` folder contains a cleaned and documented version of all the investigations conducted, including:
  + the implementation of layers for `kde_learning`, 
  + the implementation of layers with the `pseudo-attention` mechanism on past activations, 
  + the implementation of layers with the weight updating through the `hebbian`-like learning rule for fine-tuning  pre-trained models,
  + the implementation of layers with the `pseudo-attention` mechanism for fine-tuning pre-trained models
  + additionally, the folder `simulations` includes experiments with self-driving and boids flocking simulations using the pseudo-attention layer, compared to a DQN-Agent.
+ The `tests` folder contains some tests that have been conducted and promising investigations for the future.

## Installation
We report the file to create a conda environment with all the requirements.

``` bash
conda create --name <env-name> --file ./requirements.txt
```

Additionally, the notebooks which includes the simulations experiments has inside the specific version of `python`, `panda3d`, `cython`, `gym` and `pgdrive` required.

## Usage

The `d2bl` folder contains the implemented layers and further experiments proposed in the report [here](https://drive.google.com/file/d/1dBjPFGAVfrP30t2qLfGGunRPqu2ByOj-/view?usp=sharing). One of the advantages of using notebooks is that they are plug-and-play. To facilitate your exploration of the data and the methods, for each folder we have organized the code into python files `module.py` that contains different modules. You can use these files to import functions and classes that are relevant for your own research questions. Each file contains documentation and comments that explain how to use the code and what it does.


We are thrilled to share with you some of the concepts that have guided our research and development in this field. These concepts have the potential to open up new horizons and possibilities for future exploration and discovery. We hope that the ideas introduced so far can inspire novel pioneering works that will advance the state of the art and contribute to the advancement of human knowledge and well-being.


## Contributing

We have made our implementation publicly available, and we welcome anyone who would like to join us and contribute to the project.
### Contact
If you have suggestions or ideas for further improvemets please contact us.
- riccardo tedoldi: [@riccardotedoldi](https://www.instagram.com/riccardotedoldi/)
- andrea coppari: [@andreacoppari](https://it.linkedin.com/in/andreacoppari1005)

Also, if you find any bugs or issues, please let us know. We are happy to fix them!

## License
The source code for the site is licensed under the MIT license, which you can find in the LICENSE file.


<!-- ### TODO LIST
 -->

## To cite us
```bibtex
@misc{CoppariTedoldi2023,
    title   = {D2BL: Investigation on distribution-driven backpropagation learning},
    author  = {Andrea Coppari, Riccardo Tedoldi},
    year    = {2023},
    url  = {https://github.com/r1cc4r2o/D2BL}
}
```

