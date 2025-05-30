# %%
import numpy as np
import qiskit as qs
from qiskit.circuit.library import TwoLocal

import matplotlib.pyplot as plt
from qiskit_aer import AerSimulator as Aer
from qiskit import transpile
from qiskit_aer.primitives import SamplerV2
from sklearn.linear_model import LinearRegression,Ridge
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService
from sklearn.metrics import mean_squared_error,root_mean_squared_error
from scipy import stats
import time
import logging
from datetime import datetime
from multiprocessing import Pool
from matplotlib.animation import FuncAnimation
import pickle
from joblib import Parallel, delayed

from qiskit_aer.noise import (
    NoiseModel,
    QuantumError,
    ReadoutError,
    depolarizing_error,
    pauli_error,
    thermal_relaxation_error,
)

date = str(datetime.now())
seed = 1000
# from qiskit_machine_learning.circuit.library import RawFeatureVector
np.random.seed(seed)  # For reproducibility

#params
n_shots = 60000 #should be multiple of 1000

#number of memory and readout qubits
n_mem_qubits = 6
n_read_qubits = 6
context_length = 3
repeat_blocks = 3
n_qubits = n_mem_qubits + n_read_qubits



# %%
# QRC circuit definition
def quantum_reservoir(input_data, W_in, W_bias,W_hidden, W_entangle,n_mem_qubits, n_read_qubits, context_length,repeat_blocks):
    n_qubits = n_mem_qubits + n_read_qubits
    q_mem = qs.circuit.QuantumRegister(n_qubits, 'memory')
    #q_read = qs.circuit.QuantumRegister(n_read_qubits, 'readout')
    qc = qs.QuantumCircuit(q_mem, name='Reservoir')
    register_names = []
    for i in range(context_length-1,len(input_data)):
        t = []
        for j in range(context_length):
            t.append(input_data[i-j])
        t = np.array(t)
        for j in range(repeat_blocks):
            for k in range(0,n_qubits,2):
                encoding = encode_input(W_in[k],t)
                encoding2 = encode_input(W_in[k+1],t)
                phi, theta, omega = encoding
                phi2, theta2, omega2 = encoding2
                bias1, bias2, bias3 = W_bias[k]
                bias21, bias22, bias23 = W_bias[k+1]
                #qc.u(phi+ bias1, theta+ bias2, omega + bias3, k)
                #qc.u(phi2 + bias21, theta2 + bias22, omega2 + bias23, k+1)
                Rot(qc, phi=phi+ bias1, theta=theta+ bias2, omega=omega + bias3, wires=k)
                Rot(qc, phi=phi2 + bias21, theta=theta2 + bias22, omega=omega2 + bias23, wires=k+1)
                qc.cx(k,k+1)
                #qc.u(phi+ bias1, theta+ bias2, omega + bias3, k)
                #qc.u(phi2 + bias21, theta2 + bias22, omega2 + bias23, k+1)
                Rot(qc, phi=phi+ bias1, theta=theta+ bias2, omega=omega + bias3, wires=k)
                Rot(qc, phi=phi2 + bias21, theta=theta2 + bias22, omega=omega2 + bias23, wires=k+1)
                qc.cry(W_hidden[k//2,0], k, k+1)
                #qc.u(phi+ bias1, theta+ bias2, omega + bias3, k)
                #qc.u(phi2 + bias21, theta2 + bias22, omega2 + bias23, k+1)
                Rot(qc, phi=phi+ bias1, theta=theta+ bias2, omega=omega + bias3, wires=k)
                Rot(qc, phi=phi2 + bias21, theta=theta2 + bias22, omega=omega2 + bias23, wires=k+1)
                qc.crx(W_hidden[k//2,1], k, k+1)
            for k in range(0,n_qubits,2):
                qc.crz(W_entangle[k//2],k,(k+2)%n_qubits)



       #qc.barrier()
        register_names.append(f'cr_{i}')
        temp = qs.circuit.ClassicalRegister(n_read_qubits, f'cr_{i}')
        qc.add_register(temp)
        # perform measurement and reset
        ind = [i for i in range(n_qubits) if i % 2 == 1]
        qc.measure(ind, temp)
        qc.reset(ind)
        #qc.barrier()

    return qc,register_names

def encode_input(W_in, input_value):
    #np.random.seed(42)  # For reproducibility
    output = np.dot(input_value,W_in)
    return output

def Rot(qc, phi=0, theta=0, omega=0, wires=0):
    qc.rz(phi, wires)
    qc.rx(theta, wires)
    qc.rz(omega, wires)

def process_time_step(t, counts_column, num_outcomes):
    outcome_integers = np.array([int(bitstring, 2) for bitstring in counts_column])
    hist = np.bincount(outcome_integers, minlength=num_outcomes)
    return hist

def get_histogram_from_memory(counts):
    counts = np.array([shot.split() for shot in counts])
    num_shots, time_steps = counts.shape
    num_outcomes = 2 ** n_read_qubits

    results = Parallel(n_jobs=-1)(
        delayed(process_time_step)(t, counts[:, t], num_outcomes) for t in range(time_steps)
    )

    features = np.array(results) / num_shots
    features = np.flip(features, axis=0)
    return features

def get_probability_matrix(result_data, identifiers, n_read_qubits):
    # Number of possible outcomes for n qubits (2^n)
    num_outcomes = 2 ** n_read_qubits
    
    # Time steps are defined by the length of the identifiers list
    time_steps = len(identifiers)

    # Initialize the features array (time_steps, num_outcomes)
    features = np.zeros((time_steps, num_outcomes))

    # Iterate over each time step
    for t, identifier in enumerate(identifiers):
        # Dynamically access the counts using getattr
        counts = getattr(result_data, identifier).get_counts()

        # Convert bitstring outcomes to integers
        outcome_integers = np.array([int(bitstring, 2) for bitstring in counts.keys()])
        # Create a histogram of the counts
        hist = np.zeros(num_outcomes)
        for outcome, count in zip(outcome_integers, counts.values()):
            hist[outcome] = count
        
        # Normalize the histogram to get probabilities
        features[t] = hist / np.sum(hist)
    
    
    return features

# Function to set a specific percentage of weights to zero
def apply_sparsity(matrix, sparsity=0.9):
    """Sets a given percentage of the matrix weights to zero."""
    num_elements = matrix.size
    num_zero = int(sparsity * num_elements)
    
    # Randomly select indices to be set to zero
    zero_indices = np.random.choice(num_elements, num_zero, replace=False)
    
    # Flatten the matrix, set the selected indices to zero, and reshape back
    matrix_flat = matrix.flatten()
    matrix_flat[zero_indices] = 0
    return matrix_flat.reshape(matrix.shape)

    

if __name__ == '__main__':
	f = open(f'./logs/{n_qubits}_{n_shots}_qrnn_{date}_rep{repeat_blocks}_FC.txt','w')
	for i in range (1):
		#n_mem_qubits+=1
		#n_read_qubits+=1
		#n_qubits+=2
		f.write('\nSeed: ')
		f.write(str(seed))

		start = time.time()
		# %%
		#num data points
		washout = 100
		n_pts = 6900

		#Load in lorenz data
		train_data_lorenz = np.load('./data/train_data_lorenz.npy')
		test_data_lorenz = np.load('./data/test_data_lorenz.npy')

		# Split data into train and test sets
		train_data = train_data_lorenz[washout:washout+n_pts]
		test_data = test_data_lorenz

		# Use only the first component of the Lorenz system for the input signal
		train_input_signal = train_data[:, 0]
		test_input_signal = test_data[:, 0]
		#random weights from -1 to 1
		W_in = np.random.normal(loc=np.pi/(8*context_length*repeat_blocks),scale=np.pi/(24*context_length*repeat_blocks), size=(n_qubits,context_length,3))
		W_bias = np.random.normal(loc=np.pi/(12*context_length*repeat_blocks), scale=np.pi/(36*context_length*repeat_blocks), size=(n_qubits,3))
		W_hidden = np.random.normal(loc=0, scale=np.pi/6, size=(n_mem_qubits,2))
		W_entangle = np.random.normal(loc=0, scale=np.pi/30, size=(n_mem_qubits)) #Creates weak entanglement

        #Sets 90% of the weights to zero
		W_hidden = apply_sparsity(W_hidden, sparsity=0.8)
		W_entangle = apply_sparsity(W_entangle, sparsity=0.2)
      
		print('Input weights: ')
		print(W_in)
		print('Bias weights: ')
		print(W_bias)
		print("Hidden weights: ")
		print(W_hidden)
		print("Entangle weights")
		print(W_entangle)
		f.write('\nContext length: ')
		f.write(str(context_length))
		f.write('\nW_in:')
		f.write(np.array2string(W_in))
		f.write('\nW_bias')
		f.write(np.array2string(W_bias))
		f.write('\nHidden layer weights:')
		f.write(np.array2string(W_hidden))
		f.write('\nEntangle weights: ')
		f.write(np.array2string(W_entangle))
		qc, register_names = quantum_reservoir(train_input_signal, W_in, W_bias, W_hidden, W_entangle,n_mem_qubits, n_read_qubits,context_length,repeat_blocks)
		NoiseModel().from_backend()
		#figure = plt.figure(qc.draw('mpl'))
		#figure.savefig('q_rnn.pdf')
        #Need API key for IBM Qiskit Runtime if a noise model is used
		service = QiskitRuntimeService(channel="ibm_quantum",token='')
		backend = service.backend('ibm_fez')
		# Use Aer's qasm_simulator, with noise model
		backend = Aer.from_backend(backend)
		#sim = Aer(method='statevector',device='GPU',noise_model=noise_model)
		#Open sampler from pickle
		sampler = SamplerV2()
		sampler._backend = backend
		sampler._options.backend_options.update({'batched_shots_gpu': True})
		sampler.options.run_options.update({'batched_shots_gpu': True})
        
		t_qc = generate_preset_pass_manager(optimization_level=1,backend=sampler._backend).run(qc)
        #Add batched_shots_gpu to backend options
		print(sampler._options.backend_options)
		print('Depth of circuit: ')
		print(t_qc.depth())
		f.write(f'\n Depth of circuit: {qc.depth()}')

		result = sampler.run([t_qc], shots=n_shots).result()
		expectation = get_probability_matrix(result[0].data,register_names,n_read_qubits)

		np.save(f'./results_server/expectation_values/exp_values_{n_shots}_shots_{n_qubits}_qubits_rep{repeat_blocks}_{date}_{seed}', expectation)
		# %%
		#Plot counts
		plt.figure(figsize=(12, 6))
		plt.title(f'Expectation values for each readout qubit with {n_shots} shots')
		plt.xlabel('Time')
		plt.ylabel('Z Expectation Value')
		for i in range(expectation.shape[1]):
			plt.plot(expectation[:,i], label=f'Probability Amplitude {i}')
		plt.legend()
		plt.savefig(f'./results_server/qrnn_exp_values{n_qubits}_{date}_rep{repeat_blocks}_{seed}.pdf')


		condition_number = np.linalg.cond(expectation)

		f.write(f'\nCondition number of expectation matrix: {condition_number}')
		# %%


		washout_length = 300
		train_offset = (1+context_length-1)
		# Prepare training and test sets
		X_train = expectation[washout_length:-1]

		y_train = train_data[washout_length+train_offset:, 1:]  # Predict the second and third components

		# Train a ridge regression model
		ridge_regressor = Ridge(alpha=1e-8)
		ridge_regressor.fit(X_train, y_train)

		# Predict and evaluate the model
		y_train_pred = ridge_regressor.predict(X_train)
		#y_test_pred = ridge_regressor.predict(test_outputs[:-1])

		#Check RMSE
		train_rmse = root_mean_squared_error(y_train[:,0], y_train_pred[:,0])
		#test_rmse = mean_squared_error(test_data[washout_length+1:, 1:], y_test_pred, squared=False)
		f.write(f'\nTrain RMSE for Y: {train_rmse:.4f}')
		train_rmse = root_mean_squared_error(y_train[:,1], y_train_pred[:,1])
		#test_rmse = mean_squared_error(test_data[washout_length+1:, 1:], y_test_pred, squared=False)
		f.write(f'\nTrain RMSE for Z: {train_rmse:.4f}')
		train_rmse = root_mean_squared_error(y_train, y_train_pred)
		f.write(f'\nTotal RMSE for Train: {train_rmse:.4f}')



		#Plot the predicted and true second and third components of the Lorenz system
		import matplotlib.pyplot as plt
		plt.figure(figsize=(12, 6))
		plt.title(f'Train prediction of Z with {n_qubits} qubits and {n_shots} shots')
		plt.xlabel('Time')
		plt.ylabel('Normalized Value')
		plt.plot(train_data[washout_length+train_offset:, 2], label='True z')
		plt.plot(y_train_pred[:,1], label='Predicted z',linestyle='dashed')
		plt.legend()
		plt.savefig(f'./results_server/qrnn_pred_z_{n_qubits}_{date}_rep{repeat_blocks}_FC.pdf')

		plt.figure(figsize=(12, 6))
		plt.title(f'Train prediction of Y with {n_qubits} qubits and {n_shots} shots')
		plt.xlabel('Time')
		plt.ylabel('Normalized Value')
		plt.plot(train_data[washout_length+train_offset:, 1], label='True y')
		plt.plot(y_train_pred[:,0], label='Predicted y',linestyle='dashed')
		plt.legend()
		plt.savefig(f'./results_server/qrnn_pred_y_{n_qubits}_{date}_rep{repeat_blocks}_FC.pdf')


		## Start test data set evaluation ##
		#del qc
		#del sim
		#del result
		#del expectation

		qc_test,register_names_test = quantum_reservoir(test_input_signal, W_in, W_bias, W_hidden, W_entangle,n_mem_qubits, n_read_qubits,context_length, repeat_blocks)

		#figure = plt.figure(qc.draw('mpl'))
		#figure.savefig('q_rnn.pdf')

		# Use Aer's qasm_simulator
		# backend = Aer.get_backend('qasm_simulator')

		t_qc_test = generate_preset_pass_manager(optimization_level=1,backend=sampler._backend).run(qc_test)
		print('Depth of circuit: ')
		print(t_qc_test.depth())
		f.write(f'\n Depth of circuit: {qc_test.depth()}')

		result = sampler.run([t_qc], shots=n_shots).result()
		expectation = get_probability_matrix(result[0].data,register_names,n_read_qubits)

		np.save(f'./results_server/expectation_values/test_exp_values_{n_shots}_shots_{n_qubits}_qubits_rep{repeat_blocks}_{seed}', expectation)
		#Plot counts
		plt.figure(figsize=(12, 6))
		plt.title(f'Test expectation values for each readout qubit with {n_shots} shots')
		plt.xlabel('Time')
		plt.ylabel('Z Expectation Value')
		for i in range(expectation.shape[1]):
			plt.plot(expectation[:,i], label=f'Probability Amplitude {i}')
		plt.legend()
		plt.savefig(f'./results_server/qrnn_exp_values_test{n_qubits}_{date}_rep{repeat_blocks}_{seed}.pdf')


		condition_number = np.linalg.cond(expectation)

		f.write(f'\nCondition number of test expectation matrix: {condition_number}')
		# %%


		washout_length = 300
		train_offset = (1+context_length-1)
		# Prepare training and test sets
		X_train = expectation[washout_length:-1]

		y_test = test_data[washout_length+train_offset:, 1:]  # Predict the second and third components

		# Use existing model for test data
		y_test_pred = ridge_regressor.predict(X_train)
		#y_test_pred = ridge_regressor.predict(test_outputs[:-1])

		#Check RMSE
		test_rmse = root_mean_squared_error(y_test[:,0], y_test_pred[:,0])
		#test_rmse = mean_squared_error(test_data[washout_length+1:, 1:], y_test_pred, squared=False)
		f.write(f'\nTest RMSE for Y: {test_rmse:.4f}')
		test_rmse = root_mean_squared_error(y_test[:,1], y_test_pred[:,1])
		#test_rmse = mean_squared_error(test_data[washout_length+1:, 1:], y_test_pred, squared=False)
		f.write(f'\nTest RMSE for Z: {test_rmse:.4f}')
		test_rmse = root_mean_squared_error(y_test, y_test_pred)
		f.write(f'\nTotal RMSE for Test: {test_rmse:.4f}')


		#Plot the predicted and true second and third components of the Lorenz system
		plt.figure(figsize=(12, 6))
		plt.title(f'Test prediction of Z with {n_qubits} qubits and {n_shots} shots')
		plt.xlabel('Time')
		plt.ylabel('Normalized Value')
		plt.plot(test_data[washout_length+train_offset:, 2], label='True z')
		plt.plot(y_test_pred[:,1], label='Predicted z',linestyle='dashed')
		plt.legend()
		plt.savefig(f'./results_server/qrnn_pred_z_test_{n_qubits}_{date}_rep{repeat_blocks}_{seed}.pdf')

		plt.figure(figsize=(12, 6))
		plt.title(f'Test prediction of Y with {n_qubits} qubits and {n_shots} shots')
		plt.xlabel('Time')
		plt.ylabel('Normalized Value')
		plt.plot(test_data[washout_length+train_offset:, 1], label='True y')
		plt.plot(y_test_pred[:,0], label='Predicted y',linestyle='dashed')
		plt.legend()
		plt.savefig(f'./results_server/qrnn_pred_y_test_{n_qubits}_{date}_rep{repeat_blocks}_FC_{seed}.pdf')


		end = time.time()

		f.write(f'\nTime taken: {end-start:.2f} seconds')
		f.close()
          
		#### Extra code to create an animation of the outputs
		expectation = expectation[washout_length:]
		# Animation function to update the bar chart for each time step
		def update_histogram(frame, features, bar_container, bit_labels):
			histogram = features[frame]
			
			# Update the heights of the bars
			for rect, h in zip(bar_container, histogram):
				rect.set_height(h)

			plt.title(f'Histogram for Time Step {frame}')
			return bar_container

		# Setup the figure and axis for plotting
		fig, ax = plt.subplots()
		num_outcomes = expectation.shape[1]
		bit_labels = [format(i, f'0{int(np.log2(num_outcomes))}b') for i in range(num_outcomes)]

		# Initialize the bar chart with the first time step
		bars = ax.bar(bit_labels, expectation[0])

		# Labeling and rotation
		plt.xlabel('Qubit States (bitstrings)')
		plt.ylabel('Probability')
		plt.xticks(rotation=45)

		# Create the animation
		anim = FuncAnimation(
			fig, update_histogram, frames=len(expectation), fargs=(expectation, bars, bit_labels), blit=False, repeat=False,interval=100
		)


		# Show the animation
		anim.save(f'QRC_anim_{date}.gif', writer='imagemagick')



