# Linear-Distance-Metric-Learning
This repository contains code and resources related to the paper "Linear Distance Metric Learning." 
The paper explores the topic of distance metric learning and presents some novel approaches to learn a liner distance metric.

### Abstract

In linear distance metric learning, we are given data in one Euclidean metric space and the goal is to find an appropriate linear map to another Euclidean metric space which respects certain distance conditions as much as possible. In this paper, we formalize a simple and elegant method which reduces to a general continuous convex loss optimization problem, and for different noise models we derive the corresponding loss functions. We show that even if the data is noisy, the ground truth linear metric can be learned with any precision provided access to enough samples, and we provide a corresponding sample complexity bound. Moreover, we present an effective way to truncate the learned model to a low-rank model that can provably maintain the accuracy in loss function and in parameters -- the first such results of this type.  Several experimental observations on synthetic and real data sets support and inform our theoretical results.  .


### Repository Structure

The repository is structured as follows:

- [`Real Data`/](): This directory contains the implementation of the linear distance metric learning algorithm for real data. 
- [`Synthetic Data`/](): In this directory, you will find the assessment of our model on synthetic data. Additionally, it includes a comparison of our model's performance with the DML-eig model.





### BibTeX Citation

If you find this work useful or refer to it in your own research, please cite the paper using the following BibTeX entry:
```
@article{YourLastName2023Linear,
  title={Linear Distance Metric Learning},
  author={Your Last Name, Your First Name},
  journal={Journal Name},
  year={2023},
  volume={},
  number={},
  pages={},
  doi={}
}
```
