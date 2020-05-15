# Loopy Belief Propagation for Binary Image Denoising

## Theory

Loopy Belief Propagation allows us to perform approximate inference on a grid structured Markov Network. 

Using a Bethe Cluster Graph representation, the message update equations are as follows:

<p align="center">
  <img src="./images/loopy_equations.png" width="600"> 
</p>

An implementation of loopy belief propagation for binary image denoising. Both sequential and parallel updates are implemented.

For running the code, run `python3 loopy_bp.py -p PATH TO IMAGE/CSV -m MODE (seq or sync) -n ADD EXTERNAL NOISE (True or False, for testing purposes)`

