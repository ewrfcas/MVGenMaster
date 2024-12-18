# MVGenMaster

[Preprint] MVGenMaster: Scaling Multi-View Generation from Any Image via 3D Priors Enhanced Diffusion Model

[[arXiv]](https://arxiv.org/abs/2411.16157) [[Project Page]](https://ewrfcas.github.io/MVGenMaster/)

### Abstract

We introduce **MVGenMaster**, a multi-view diffusion model enhanced with 3D priors to address versatile Novel View Synthesis (NVS) tasks. MVGenMaster leverages 3D priors that are warped using metric depth and camera poses, significantly enhancing both generalization and 3D consistency in NVS.
Our model features a simple yet effective pipeline that can generate up to 100 novel views conditioned on variable reference views and camera poses with a single forward process.
Additionally, we have developed a comprehensive large-scale multi-view image dataset called **MvD-1M**, comprising up to 1.6 million scenes, equipped with well-aligned metric depth to train MVGenMaster.
Moreover, we present several training and model modifications to strengthen the model with scaled-up datasets.
Extensive evaluations across in- and out-of-domain benchmarks demonstrate the effectiveness of our proposed method and data formulation.
Models and codes will be released soon.