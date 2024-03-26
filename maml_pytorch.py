import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.autograd import grad


def mlp(x, params):
    h = torch.relu(torch.nn.functional.linear(x, params[0], bias=params[1]))
    h = torch.relu(torch.nn.functional.linear(h, params[2], bias=params[3]))
    return torch.nn.functional.linear(h, params[4], bias=params[5])


class Task:

    def __init__(self, a=None, b=None, slope=None, bias=None, dimension = None):
        self.a = a
        self.b = b
        self.slope = slope
        self.bias = bias
        self.dimension = dimension

    def sample_sinusoid(self, K):
        '''Sample K sinusoid data points from the task. Return x, y and the loss function.'''
        x = torch.rand((K, 1)) * 10 - 5  # Sample x in [-5, 5]
        y = self.a * torch.sin(x + self.b)
        loss_fct = nn.MSELoss()
        return x, y, loss_fct
    
    def sample_linear(self,N):
        '''Sample K linear data points from the task. Return x, y and the loss function.'''
        x = torch.randn((N, self.dimension))
        y = torch.matmul(x, self.slope) + self.bias
        loss_fct = nn.MSELoss()
        return x, y, loss_fct


@torch.no_grad() # Decorator to disable gradient computation
def sample_task(functional_class = "sinusoid", linear_bias = False, bias_var = 1, dimension = 10):
    if functional_class == "sinusoid":
        # for sinusoid tasks
        a = torch.rand(1).item() * 4.9 + .1  # Sample the amplitude in [0.1, 5.0]
        b = torch.rand(1).item() * np.pi  # Sample the phase in [0, pi]
        return Task(a=a, b=b)
    elif functional_class == "linear":
        # for linear tasks
        slope = torch.randn(dimension, 1) + 1 if dimension > 1 else torch.randn(dimension).item() + 1
        if linear_bias:
            bias = torch.rand(1) * torch.sqrt(torch.tensor(bias_var)).item()
        else:
            bias = 0
        return Task(slope=slope, bias= bias, dimension = dimension)


def perform_k_training_steps(params, task, N, inner_training_steps, alpha, functional_class='sinusoid', device='cpu'):
    ''' Perform k gradient steps on the task (inner loop)
    :param params: List of parameters to optimize
    :param task: Task to perform the gradient steps on
    :param N: Batch size
    :param inner_training_steps: Number of gradient steps
    :param alpha: Learning rate
    :param functional_class: Type of function to approximate
    :param device: Device to run the computations on
    :return: Updated parameters
    '''
    for epoch in range(inner_training_steps):
        if functional_class == 'sinusoid':
            x_batch, target, loss_fct = task.sample_sinusoid(N)
        elif functional_class == 'linear':
            x_batch, target, loss_fct = task.sample_linear(N)

        loss = loss_fct(mlp(x_batch.to(device), params), target.to(device))

        for p in params:  # Zero grad
            p.grad = None
        gradients = grad(loss, params)
        for p, g in zip(params, gradients):  # Grad step
            p.data -= alpha * g
    return params


def maml(p_model, meta_optimizer, inner_training_steps, nb_epochs, N, alpha, M=10, functional_class="sinusoid", linear_bias = False, bias_var = 1, dimension = 10, device='cpu'):
    """
    Algorithm from https://arxiv.org/pdf/1703.03400v3.pdf (MAML for Few-Shot Supervised Learning)
    :param p_model: List of parameters to optimize
    :param meta_optimizer: Optimizer for the meta-update
    :param inner_training_steps: Number of gradient steps in the inner loop
    :param nb_epochs: Number of epochs for the outer loop
    :param N: Batch size for the inner loop
    :param alpha: Learning rate for the inner loop
    :param M: Number of tasks to sample
    :param functional_class: Type of function to approximate
    :param linear_bias: Whether to include a bias term in the linear function
    :param bias_var: Variance of the bias term in the linear function
    :param dimension: Dimension of the input space in the linear function
    :param device: Device to run the computations on
    :return: List of training losses
    """
    training_loss = []
    for epoch in tqdm(range(nb_epochs)):  # Line 2 in the pseudocode

        theta_i_prime = []
        D_i_prime = []

        # Sample batch of tasks
        tasks = [sample_task(functional_class=functional_class, linear_bias=linear_bias, bias_var=bias_var, dimension=dimension) for _ in range(M)]  # Line 3 in the pseudocode
        for task in tasks:
            theta_i_prime.append(
                perform_k_training_steps(
                    params=[p.clone() for p in p_model],
                    task=task,
                    N=N,
                    inner_training_steps=inner_training_steps,
                    alpha=alpha,
                    functional_class=functional_class,
                    device=device
                )
            )
            # Sample data points Di' for the meta-update (line 8 in the pseudocode)
            if functional_class == 'sinusoid':
                x, y, loss_fct = task.sample_sinusoid(25)
            elif functional_class == 'linear':
                x, y, loss_fct = task.sample_linear(25)
            D_i_prime.append((x, y, loss_fct))

        # Meta update
        meta_optimizer.zero_grad()
        batch_training_loss = []
        for i in range(M):
            x, y, loss_fct = D_i_prime[i]
            f_theta_prime = theta_i_prime[i]
            # Compute \nabla_theta L(f_theta_i_prime) for task ti
            loss = loss_fct(mlp(x.to(device), f_theta_prime), y.to(device))
            loss.backward()
            batch_training_loss.append(loss.item())

        meta_optimizer.step()  # Line 10 in the pseudocode
        training_loss.append(np.mean(batch_training_loss))
    
    # Save the model
    if linear_bias:
        torch.save({
            'model_state_dict': [p.clone() for p in p_model],
            'meta_optimizer_state_dict': meta_optimizer.state_dict(),
        }, f'models/MAML_M{M}_N{N}_dim{dimension}_bias{bias_var}.pth')
    else:
        torch.save({
            'model_state_dict': [p.clone() for p in p_model],
            'meta_optimizer_state_dict': meta_optimizer.state_dict(),
        }, f'models/MAML_M{M}_N{N}_dim{dimension}.pth')

    return training_loss


if __name__ == "__main__":
    device = 'cpu'
    functional_class = 'linear'
    linear_bias = True
    bias_var = 2
    dimension = 10
    M = 10
    N = 10
    params = [torch.rand(40, dimension, device=device).uniform_(-np.sqrt(6. / 41), np.sqrt(6. / 41)).requires_grad_(),
              torch.zeros(40, device=device).requires_grad_(),
              torch.rand(40, 40, device=device).uniform_(-np.sqrt(6. / 80), np.sqrt(6. / 80)).requires_grad_(),
              torch.zeros(40, device=device).requires_grad_(),
              torch.rand(1, 40, device=device).uniform_(-np.sqrt(6. / 41), np.sqrt(6. / 41)).requires_grad_(),
              torch.zeros(1, device=device).requires_grad_()]

    device = 'cpu'
    meta_optimizer = torch.optim.Adam(params, lr=1e-3)
    training_loss = maml(
        p_model= params,
        meta_optimizer= meta_optimizer,
        inner_training_steps=1,
        nb_epochs=70_000,
        N=N,
        alpha=1e-3,
        M=M,
        functional_class=functional_class,
        linear_bias=linear_bias,
        bias_var=bias_var,
        device=device, 
        dimension=dimension
    )

    # # Evaluation process
    # plt.title('MAML, K=10')
    # x = torch.linspace(-5, 5, 50).to(device)
    # y = mlp(x[..., None], params)
    # plt.plot(x.data.cpu().numpy(), y.data.cpu().numpy(), c='lightgreen', linestyle='--', linewidth=2.2,
    #          label='pre-update')
    # # New task
    # task = sample_task(functional_class=functional_class, linear_bias=linear_bias, bias_var=bias_var, dimension = dimension)
    # if functional_class == 'linear':
    #     ground_truth_y = task.slope * x + task.bias
    # elif functional_class == 'sinusoid':
    #     ground_truth_y = task.a * torch.sin(x + task.b)
    # plt.plot(x.data.cpu().numpy(), ground_truth_y.data.cpu().numpy(), c='red', label='ground truth')
    # # Fine-tuning, 10 gradient steps
    # new_params = perform_k_training_steps(
    #     params = [p.clone() for p in params], 
    #     task = task, 
    #     batch_size=10,
    #     inner_training_steps=10,
    #     alpha=1e-3,
    #     functional_class=functional_class,
    #     device=device
    # )
    # # After 10 gradient steps
    # y = mlp(x[..., None], new_params)
    # plt.plot(x.data.cpu().numpy(), y.data.cpu().numpy(), c='darkgreen', linestyle='--', linewidth=2.2,
    #          label='10 grad step')
    # plt.legend()
    # plt.savefig('maml.png')
    # plt.show()