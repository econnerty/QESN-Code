1. Run QESN_gpu_sim.py to collect simulated results. The code is set to only run a train data set, but it must run a train and test set to do the full predictions. The results can be split into a train and test set after running the experiment.
2. Run predictions_from_probs.ipynb to generate predictions of the Lorenz system with the train and test measurements collected from the circuit.
3. The other files are for running on the QPU and require an IBM token.
