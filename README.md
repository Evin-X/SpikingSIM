## ***SpikingSIM***: **A Bio-inspired Spiking Simulator**

A simulator of spiking cameras that converts images/videos to spiking data.


#### __$\bullet$ Introduction__

 
The motivation of this work is to simulate the working principle of spiking cameras to convert existing image/video data to spiking data.

As shown in the Figure 1 bellow, we formulate the relation: grayscale (image) -> scene brightness intensity -> spike firing rate (spike). The source and distribution of noises are also considered to simulate more realistic spiking data.
<div align=center>
<img alt="Figure 1" width="90%" src="https://github.com/Evin-X/SpikingSIM/blob/main/Figure/Figure%201.png"/>
</div>

In the following example, we first reconstruct intensity map (a) from raw spikes (c) and use it to simulate spikes (d). The image (b) is reconstructed from simulated spikes (d). The visualization shows that the synthetic data (d) generated by *SpikingSIM* are quite similar with the raw data (c). (It might take a little time for the images to load.) 
<div align=center>
<img alt="Figure 2" width="60%" src="https://github.com/Evin-X/SpikingSIM/blob/main/Figure/Figure%202.gif"/>
</div>

We also convert an image (from Caltech-101 dataset) into synthetic spikes and visualize the results as bellow.
<div align=center>
<img alt="Figure 3" width="90%" src="https://github.com/Evin-X/SpikingSIM/blob/main/Figure/Figure%203.gif"/>
</div>

#### __$\bullet$ Usage__

Code is in ./Source/`sim.py`.

1. Input: Images (`load_image`, $G$) | Simulation Time (`sim_time`, $T$) 

2. Output: Spike Array (`syn_spike`, $H\times W\times T$) 

3. Noise: Noise intensity can be adjusted by Gaussian (`Inherent_Noise`, $\mathcal{N}$) and Poisson (`Diffuse_Noise`, $\mathcal{P}$) models

4. Simulation: Simulate the spike generation process of spiking cameras (`Simulation`)

   \* Note: We release a *simplified version* for fast simulation (`Simulation_Simplified`)

#### __$\bullet$ Citation__

We hope this work could promote your research. Thanks.

_Zhao J, Zhang S, Ma L, Yu Z and Huang T. 2022. SpikingSIM: A Bio-inspired Spiking Simulator // 2022 IEEE International Symposium on Circuits and Systems (ISCAS). IEEE, 2022._
