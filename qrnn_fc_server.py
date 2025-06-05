# %%
import numpy as np
import qiskit as qs
from qiskit.circuit.library import TwoLocal

import matplotlib.pyplot as plt
from qiskit_aer import AerSimulator as Aer
from qiskit import transpile,qpy
from qiskit_aer.primitives import SamplerV2
from qiskit_aer.noise import NoiseModel
from sklearn.linear_model import LinearRegression,Ridge
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService
from sklearn.metrics import mean_squared_error,root_mean_squared_error
from scipy import stats
import time
import logging
import QRNN as QRNN
from datetime import datetime
from multiprocessing import Pool
from matplotlib.animation import FuncAnimation
import pickle
from joblib import Parallel, delayed
from qiskit.circuit.library import XGate, YGate
from qiskit.transpiler import InstructionProperties
from qiskit.transpiler import PassManager
from qiskit.circuit.equivalence_library import (
    SessionEquivalenceLibrary as sel,
)
from qiskit.transpiler.passes import BasisTranslator
from qiskit.transpiler.passes.scheduling import (
    ALAPScheduleAnalysis,
    ASAPScheduleAnalysis,
    PadDynamicalDecoupling,
)

from qiskit_aer.noise import (
    NoiseModel,
    QuantumError,
    ReadoutError,
    depolarizing_error,
    pauli_error,
    thermal_relaxation_error,
)

date = str(datetime.now())
seed = 123123
# from qiskit_machine_learning.circuit.library import RawFeatureVector
#np.random.seed(seed)  # For reproducibility

#params
n_shots = 60000 #should be multiple of 1000

#number of memory and readout qubits
n_mem_qubits = 6
n_read_qubits = 6
context_length = 3
mem_sparsity = 0.5
ent_sparsity = 0.4
#repeat_blocks = 5
n_qubits = n_mem_qubits + n_read_qubits


if __name__ == '__main__':
	#Create loop for the different datasets
	signal_types = ['step', 'ramp',  'sinusoid']
	#signal_types = ['impulse']
	repeat_block = [1,2,3,4,5]
	for repeat_blocks in repeat_block:
		for signal_type in signal_types:
			f = open(f'./logs/{signal_type}_{n_qubits}_{n_shots}_qrnn_{date}_rep{repeat_blocks}_FC.txt','w')
			for i in range (1):
				f.write('\nSeed: ')
				f.write(str(seed))

				start = time.time()
				# %%
				#num data points
				washout = 50

				#Load in lorenz data
				#Get data
				start = 500
				n_pts = 2000
				test_pts = 500
				#train_input_signal, test_input_signal, train_data, test_data = QRNN.get_lorenz_data(startup=start,n_pts=n_pts,test_pts=test_pts)
				train_input_signal = np.load(f'./data/{signal_type}_signal.npy')
				# #Get circuit
				qc, register_names = QRNN.QRNN(train_input_signal,n_qubits=n_qubits,context_length=context_length,repeat_blocks=repeat_blocks,ent_sparsity=ent_sparsity,mem_sparsity=mem_sparsity,seed=seed)
				#random weights from -1 to 1


				#figure = plt.figure(qc.draw('mpl'))
				#figure.savefig('q_rnn.pdf')
				# Use Aer's qasm_simulator
				#noise_model = NoiseModel().from_backend(backend)
				#backend = Aer.from_backend(backend)
				backend = Aer(method='statevector',device='GPU')
				#sim = Aer(method='statevector',device='GPU',noise_model=noise_model)
				#Open sampler from pickle
				sampler = SamplerV2(options=dict(backend_options=dict(device="GPU")))
				#sampler._backend = backend
				#sampler._options.backend_options.update({'batched_shots_gpu': True,'device':'GPU'})
				sampler.options.run_options.update({'batched_shots_gpu': True})
				
				#t_qc = generate_preset_pass_manager(optimization_level=3,backend=backend).run(qc)

				#Try dynamical decoupling
				t_qc = generate_preset_pass_manager(optimization_level=3,backend=backend).run(qc)
		
				# with open(f"exp_values_{n_shots}_shots_{n_qubits}_qubits_rep{repeat_blocks}_{date}_{seed}_{start}_{n_pts}_hasdelay_8000.qpy", "wb") as file:
				# 		qpy.dump(qc, file)


				#Add batched_shots_gpu to backend options
				print(sampler._options.backend_options)
				print('Depth of circuit: ')
				print(t_qc.depth())
				f.write(f'\n Depth of circuit: {qc.depth()}')
				print('OPS:')
				print(t_qc.count_ops())
				result = [sampler.run([t_qc], shots=n_shots).result()[0].data]
				expectation = QRNN.get_probability_matrix(result, register_names, n_qubits//2)

				np.save(f'./results_server/expectation_values/{signal_type}_prob_values_{n_shots}_shots_{n_qubits}_qubits_rep{repeat_blocks}_{date}_{seed}_{start}_{n_pts}_reptest_sparse_norotationread_cnoto', expectation)
			
				f.close()



