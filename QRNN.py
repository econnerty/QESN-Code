# QRNN Implementation
# Authors: Erik Connerty and Ethan Evans
# Date: July 8th, 2024

import numpy as np
import qiskit as qs

#number of memory and readout qubits
def QRNN(input_data,n_qubits,context_length,repeat_blocks,ent_sparsity,mem_sparsity,seed):

    #Set seed
    np.random.seed(seed)

    #Assert n_qubits is even and print
    assert n_qubits % 2 == 0, "n_qubits must be even"


    #Create weights
    W_in = np.random.normal(loc=np.pi/(8*context_length*repeat_blocks),scale=np.pi/(24*context_length*repeat_blocks), size=(n_qubits,context_length,3))

    W_bias = np.random.normal(loc=np.pi/(12*context_length*repeat_blocks), scale=np.pi/(36*context_length*repeat_blocks), size=(n_qubits,3))
    W_hidden = np.random.normal(loc=0, scale=np.pi/(2*repeat_blocks), size=(n_qubits//2,2))
    W_entangle = np.random.normal(loc=0, scale=np.pi/(10*repeat_blocks), size=(n_qubits//2)) #Creates weak entanglement

    #Sets 90% of the weights to zero
    W_hidden = apply_sparsity(W_hidden, sparsity=ent_sparsity)
    W_entangle = apply_sparsity(W_entangle, sparsity=mem_sparsity)

    #Patch for fractional gates
    #W_hidden = np.abs(W_hidden)
    #W_entangle = np.abs(W_entangle)
    print('Input weights: ')
    print(W_in)
    print('Bias weights: ')
    print(W_bias)
    print("Hidden weights: ")
    print(W_hidden)
    print("Entangle weights")
    print(W_entangle)
    print('crx dynamic')
    # W_in = np.random.normal(loc=np.pi/(8*context_length*repeat_blocks),scale=np.pi/(24*context_length*repeat_blocks), size=(n_qubits,context_length,3))
    # W_bias = np.random.normal(loc=np.pi/(12*context_length*repeat_blocks), scale=np.pi/(36*context_length*repeat_blocks), size=(n_qubits,3))
    # W_hidden = np.random.normal(loc=0, scale=np.pi/6, size=(n_qubits//2,2))
    # W_hidden[np.random.rand(*W_hidden.shape) < .3] = 0  # Set sparsity


    # #W_hidden[np.abs(W_hidden) < ((np.pi)*.25)] = 0 #50% sparsity
    # W_entangle = np.random.normal(loc=0, scale=np.pi/30, size=(n_qubits//2)) #Creates weak entanglement
    # W_entangle[np.random.rand(*W_entangle.shape) < .5] = 0  # Set sparsity

    #Create quantum circuit
    q_mem = qs.circuit.QuantumRegister(n_qubits, 'memory')
    #q_read = qs.circuit.QuantumRegister(n_read_qubits, 'readout')
    qc = qs.QuantumCircuit(q_mem, name='QRNN')


    #Get odd indices for measure and reset
    ind = [i for i in range(n_qubits) if i % 2 == 1]

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

        qc.barrier()
        register_names.append(f'cr_{i}')
        temp = qs.circuit.ClassicalRegister(n_qubits//2, f'cr_{i}')
        qc.add_register(temp)
        # perform measurement and reset

        qc.measure(ind, temp)
        #qc.delay(6000) #add delay 64000dt #16000 is 64 microseconds on marrakesh #PUTTING DELAY AFTER RESET IS A BAD IDEA!
        qc.reset(ind)
        qc.barrier()
        
    return qc, register_names

def encode_input(W_in, input_value):
    output = np.dot(input_value,W_in)
    return output

def Rot(qc, phi=0, theta=0, omega=0, wires=0):
    # qc.rz(phi, wires)
    # qc.ry(theta, wires)
    # qc.rz(omega, wires)
    qc.rz(phi, wires)
    qc.rx(theta, wires)
    qc.rz(omega, wires)


def get_lorenz_data(startup=100, n_pts=1000,test_pts=500):

    #Load in lorenz data
    train_data_lorenz = np.load('./data/train_data_lorenz.npy')
    #test_data_lorenz = np.load('./data/test_data_lorenz.npy')

    # Split data into train and test sets
    train_data = train_data_lorenz[startup:startup+n_pts]
    test_data = train_data_lorenz[startup+n_pts:(startup+n_pts)+test_pts]#test_data_lorenz

    # Use only the first component of the Lorenz system for the input signal
    train_input_signal = train_data[:, 0]
    test_input_signal = test_data[:, 0]

    return train_input_signal, test_input_signal, train_data, test_data

def get_expectations_from_memory(shots):
    shots = np.array([list(map(int,word)) for item in shots for word in item.split()]).reshape(len(shots), -1, len(shots[0].split()[0]))
    expectation = np.mean(shots, axis=0)
    expectation = np.flip(expectation, axis=0)
    return expectation

# Function to set a specific percentage of weights to zero
def apply_sparsity(matrix, sparsity=0.9):
    num_elements = matrix.size
    num_zero = int(sparsity * num_elements)
    
    # Randomly select indices to be set to zero
    zero_indices = np.random.choice(num_elements, num_zero, replace=False)
    
    # Flatten the matrix, set the selected indices to zero, and reshape back
    matrix_flat = matrix.flatten()
    matrix_flat[zero_indices] = 0
    return matrix_flat.reshape(matrix.shape)

def get_probability_matrix(result_data, identifiers, n_read_qubits):
    # Number of possible outcomes for n qubits (2^n)
    num_outcomes = 2 ** n_read_qubits
    
    # Time steps are defined by the length of the identifiers list
    time_steps = len(identifiers)

    # Initialize the features array (time_steps, num_outcomes)
    results = []
    for result in result_data:
        features = np.zeros((time_steps, num_outcomes))
        # Iterate over each time step
        for t, identifier in enumerate(identifiers):
            # Dynamically access the counts using getattr
            counts = getattr(result, identifier).get_counts()

            # Convert bitstring outcomes to integers
            outcome_integers = np.array([int(bitstring, 2) for bitstring in counts.keys()])
            # Create a histogram of the counts
            hist = np.zeros(num_outcomes)
            for outcome, count in zip(outcome_integers, counts.values()):
                hist[outcome] = count
            
            # Normalize the histogram to get probabilities
            features[t] = hist / np.sum(hist)
        results.append(features)
    
    results = np.array(results)
    return np.mean(results, axis=0)


def probs_to_expectation(matrix, n_read_qubits):
    num_basis_states = 2**n_read_qubits
    assert matrix.shape[1] == num_basis_states, "Number of columns must be 2^n"

    # Precompute binary representation of basis states
    basis_states = np.arange(num_basis_states)
    binary_basis = ((basis_states[:, None] >> np.arange(n_read_qubits)) & 1)

    # Compute expectation values for each qubit
    expectation_matrix = np.zeros((matrix.shape[0], n_read_qubits))
    for i in range(n_read_qubits):
        z_contributions = 1 - 2 * binary_basis[:, i]  # Maps 0 -> +1, 1 -> -1
        expectation_matrix[:, i] = np.dot(matrix, z_contributions)
    
    return expectation_matrix

