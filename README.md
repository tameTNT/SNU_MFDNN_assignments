# Mathematical Foundations of Deep Neural Networks (MFDNN)

This repository contains the original problems alongside my submissions for Mathematical Foundations of Deep Neural Networks [M1407.001200] 
taken at Seoul National University (SNU) in the Spring semester of 2024.

All material (assignments themselves, solutions, starter code) is credited to Ernest K. Ryu (https://ernestryu.com/) and available freely on his website.

## Included files

`.` contains the problem sheets themselves alongside the starter code provided for each problem.

`./hw_workings` contains a Jupyter notebook with **my own submissions** for the programming questions of each assignment (`hw*_code.ipynb`) alongside a `pdf` version of their output (`hw*_code_output.pdf`).
Also included is a `pdf` of the handwritten workings for the mathematical problems (`hw*_workings.pdf`).

`./hw_solutions` contains the solution files organised by assignment number.

Note that dataset files are not included (including `mnist`, `cifar10`, and `kmnist`). 
The given path variables in the notebooks should be edited to a local copy of these datasets or simply redownloaded using the provided pyTorch datasets loaders (with `download=True`)

## Dependencies
All of my own notebooks were originally created and run on MacOS (arm) with the following versions. `mps` is used as the device for `pytorch` where applicable.
```
Python version: 3.11.8 | packaged by conda-forge | (main, Feb 16 2024, 20:49:36) [Clang 16.0.6 ]
Numpy version: 1.26.4
Matplotlib version: 3.8.0
PyTorch version: 2.2.1
```
