import numpy as np

# Import PyTorch
import torch
from torch.autograd import Variable
import torch
import copy
import tensorly as tl
import os
import time
import math
import csv
# Import TensorLy
import tensorly as tl
from tensorly.tucker_tensor import tucker_to_tensor
# from tensorly.random import check_random_state
from sklearn.utils import check_random_state
tl.set_backend('pytorch')

def tucker_gradient_descent(parameters_list, mixing_matrix, lrep, lret, original_tensor, iteration, level, device, penalty=0.1):
    """
    Perform one iteration of decentralized gradient descent for Tucker decomposition.

    Parameters:
    - parameters_list: dict for each agent's parameters
    - mixing_matrix: Mixing matrix for the decentralized system
    - lr: Learning rate
    - original_tensor: Original tensor A
    - penalty: Regularization penalty (default is 0.1)

    Returns:
    - Updated parameters
    - Loss value for monitoring
    """

    old_parameters_list = copy.deepcopy (parameters_list)  
    
    if iteration % 2 == 0:
        quantized_parameters_list = quantizer (parameters_list, level, device)  
    else:
        quantized_parameters_list = quantizer_1 (parameters_list, level, device)  
   
    # summation_temp = {}
    loss_value = []# List to store losses for each agent
    
    
    for i in range(num_agents):  # Number of agents

        parameters = parameters_list[i]  # ith: [core, factor0, factor1, ...]
        summation_temp = [torch.zeros_like(param) for param in parameters]   # ith: [core, factor0, factor1, ...]
        gradients = [torch.zeros_like(param) for param in parameters]
        # Consensus

        for j in range(len(parameters)):
            # parameters[j] = parameters[j] * 0
            for k in range(num_agents):
                # print(agent_tensors[j])
                summation_temp[j] += lrep * mixing_matrix[i, k] * quantized_parameters_list[k][j]  # ith agent in dict, jth tensor(core or factors), k neighbors

        # Extract core and factors from parameters
        core, factors = parameters[0], parameters[1:]  # ith core and factors.

        # Reconstruct the tensor from the decomposed form
        reconstructed_tensor = tucker_to_tensor((core, factors))

        # Compute the loss (squared L2 loss)
        loss = tl.norm(reconstructed_tensor - original_tensor, 2)

        # Add regularization penalty to the loss
        for factor in factors:
            loss = loss + penalty * torch.sum(factor.pow(2))

        # Perform backward pass and update gradients
        loss.backward()

        # Update parameters using local gradient descent
        # with torch.no_grad():
        #     for param, grad in zip(parameters, gradients):
        #         grad.copy_(-lr * param.grad)  # Copy the negative gradient to the gradients list
        #         # Manually zero the gradients after updating
        #         param.grad.zero_()
        # print(f"summation_temp:{summation_temp}")
        # print(f"gradients:{gradients}")
        # parameters_list[i] = summation_temp + gradients
        
        
        # Update parameters using local gradient descent
        for param, grad, summation in zip(parameters, gradients, summation_temp):
            grad.copy_(- lret * param.grad)  # Copy the negative gradient to the gradients list
            # Manually zero the gradients after updating
            param.grad.zero_()

            # Replace param values with the sum of grad and summation_temp
            for p, g, s in zip(parameters_list[i] , gradients, summation_temp):
                p.data = (1-lrep)*p.data+(g.data + s.data)
        
        loss_value.append(loss.item())  # Append the loss for the current agent
        # print(f"summation_temp:{summation_temp}")
        # print(f"gradients:{gradients}")
        # print(f"parameters_list[{i}]:{parameters_list[i]}")
    
    return parameters_list,  loss_value


def initialize_agent_parameters(agent_id, tensor, ranks):
    core = tl.tensor(rng.random_sample(ranks), device=device, requires_grad=True)
    factors = [tl.tensor(rng.random_sample((tensor.shape[i], ranks[i])),
                         device=device, requires_grad=True) for i in range(tl.ndim(tensor))]

    return [core]+factors

# Quantizer
def quantizer (parameters_list, level, device):  
    quantized_parameters_list = copy.deepcopy (parameters_list) 
    #print(len(quantized_parameters_list))=#agents
    for i in range(len(quantized_parameters_list)):
        # print(len(quantized_parameters_list[i]))=4, core+factors0123
        for j in range(len(quantized_parameters_list[i])):
            norm = tl.norm(parameters_list[i][j].data, 2)
            # print(quantized_parameters_list[0][0].data)
            # print(norm)
            if norm <= 0.1: #norm cannot be zero
                norm = 0.1
            level_float = level * tl.abs(parameters_list[i][j].data) / norm
            # print(level_float)
            previous_level = torch.floor(level_float)
            # print(quantized_parameters_list[i][j].data.shape)  
            is_next_level = torch.rand(*parameters_list[i][j].data.shape, device=device) < (level_float - previous_level)
            new_level = previous_level + is_next_level
            quantized_parameters_list[i][j].data = tl.sign(parameters_list[i][j].data) * norm * new_level / level   
            
    return quantized_parameters_list


def quantizer_1(parameters_list, level, device):   # d quantization level
    quantized_parameters_list = copy.deepcopy (parameters_list) 
    for i in range(len(quantized_parameters_list)):
        # print(len(quantized_parameters_list[i]))=4, core+factors0123
        for j in range(len(quantized_parameters_list[i])):
            norm = tl.norm(quantized_parameters_list[i][j].data, 2)
            # print(quantized_parameters_list[0][0].data)
            # print(norm)
            if norm <= 0.1: #norm cannot be zero
                norm = 0.1
            level_float_1 = 2 * level * quantized_parameters_list[i][j].data / norm   
            # print(level_float)
            previous_level_1 = torch.floor(level_float_1)
            previous_level_2 = torch.where(previous_level_1 % 2 == 0, previous_level_1 - 1, previous_level_1)
            # print(quantized_parameters_list[i][j].data.shape)  
            is_next_level_1 = (2*torch.rand(*quantized_parameters_list[i][j].data.shape, device=device)) < (level_float_1 - previous_level_2)
            new_level_1 = previous_level_2 + (2*is_next_level_1)
            quantized_parameters_list[i][j].data = norm * new_level_1 / (2*level)   
 
    
    return quantized_parameters_list


# Stepsize strategy   
def lr(t_list, iteration, initial_lr, initial_lr_1, regularizer, lr_alpha, lr_beta, lr_ck):
    # p = self.params
    t = iteration
    
    if t <= t_list[0]:
        lrep = initial_lr / (1 + regularizer * (t) ** lr_alpha)
        lret = initial_lr_1 / (1 + regularizer * (t) ** lr_beta)
    elif t > t_list[-2] and t < t_list[-1]:
        lrep = initial_lr / (1 + regularizer * (t_list[-2]) ** lr_alpha)
        lret = initial_lr_1 / (1 + regularizer * (t_list[-2]) ** lr_beta)
    elif t >= t_list[-1]:
        lrep = initial_lr / (1 + regularizer * (t) ** lr_alpha)
        lret = initial_lr_1 / (1 + regularizer * (t) ** lr_beta)
        pre_idx = t_list[-1] + math.ceil(lr_ck/(initial_lr / (1 + regularizer * (t) ** lr_alpha)))
        t_list.append(pre_idx)
        
    return t_list, lrep, lret


def log_to_csv(filename, iteration, loss_value, rec_error):
    """
    Log iteration, loss value, and reconstruction error to a CSV file.

    Parameters:
    - filename: Name of the CSV file
    - iteration: Current iteration
    - loss_value: Loss value
    - rec_error: Reconstruction error
    """
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file, lineterminator = '\n')
        writer.writerow([iteration])
        writer.writerow(loss_value)
        writer.writerow([rec_error.item()])
        writer.writerow([])

    

random_state = 1234
rng = check_random_state(random_state)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# shape = [10, 10, 10]
loaded_data = np.load("data/observed.npy")

# Convert the NumPy array to a PyTorch tensor using tensorly
original_tensor = tl.tensor(loaded_data, device=device, requires_grad=False)


# Initialize parameters for each agent
num_agents = 5   
ranks = [5, 5, 5]
# core = tl.tensor(rng.random_sample(ranks), device=device, requires_grad=True)
# factors = [tl.tensor(rng.random_sample((tensor.shape[i], ranks[i])),
#                  device=device, requires_grad=True) for i in range(tl.ndim(original_tensor))]

agents_parameters = [initialize_agent_parameters(i, original_tensor, ranks) for i in range(num_agents)]
# print(agents_parameters[4][3])
# Mixing matrix
agent_matrix = np.array([[0.6, 0, 0, 0.4, 0], [0.2, 0.8, 0, 0, 0], [0.2, 0.1, 0.4, 0, 0.3], [0, 0, 0, 0.6, 0.4],
                         [0, 0.1, 0.6, 0, 0.3]])

# Decentralized tensor decomposition
n_iterations = 100001
# lr = 0.0001
# initial lr
t_0 = 100
lr_ck=80
initial_lr=0.03
initial_lr_1=0.003
regularizer=0.3
lr_alpha=0.6
lr_beta=0.9
t_list = [t_0,]
pre_idx = t_0 + math.ceil(lr_ck/(initial_lr / (1 + regularizer * t_0 ** lr_alpha)))
t_list.append(pre_idx)
lrep_list = []
lret_list = []

level = 64

cwd = os.getcwd() # Get the current working directory
loca=time.strftime('%Y-%m-%d-%H-%M-%S')

results_path = os.path.join(cwd, "results") # Construct the path for the "results" directory
filename = os.path.join(results_path, f"A:QGD_{str(loca)}_I{n_iterations}_t_0:{t_0}_lrep:{initial_lr}_lret:{initial_lr_1}_lrck:_{lr_ck}_alpha:{lr_alpha}_beta:{lr_beta}_level:{level}.csv") 
print(filename)   
    
if not os.path.isdir(results_path): # Check if the "results" directory does not exist
    os.mkdir(results_path)   # Create the "results" directory if it doesn't exist
       
for iteration in range(n_iterations):
    # update lr
    t_list, lrep, lret = lr (t_list, iteration, initial_lr, initial_lr_1, regularizer, lr_alpha, lr_beta, lr_ck)  
    agents_parameters, loss_value = tucker_gradient_descent(agents_parameters, agent_matrix, lrep, lret, original_tensor, iteration, level, device)
    # agents_parameters, loss_value = tucker_gradient_descent(agents_parameters, agent_matrix, lrep, lret, original_tensor, iteration, device)
    # print(f"Iteration {iteration + 1}/{n_iterations}, Loss: {loss_value}")
    if iteration % 1000 == 0:
        print(loss_value)
        rec_error = torch.mean(torch.tensor(loss_value)) /tl.norm(original_tensor.data, 2)
        print("Epoch {},. Rec. error: {}".format(iteration, rec_error))
        log_to_csv(filename, iteration, loss_value, rec_error)
print("t_list:{}".format(t_list))
    