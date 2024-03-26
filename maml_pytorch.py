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

    def __init__(self, a=None, b=None, slope=None, bias=None):
        self.a = a
        self.b = b
        self.slope = slope
        self.bias = bias

    def sample_sinusoid(self, K):
        '''Sample K sinusoid data points from the task. Return x, y and the loss function.'''
        x = torch.rand((K, 1)) * 10 - 5  # Sample x in [-5, 5]
        y = self.a * torch.sin(x + self.b)
        loss_fct = nn.MSELoss()
        return x, y, loss_fct
    
    def sample_linear(self,K):
        '''Sample K linear data points from the task. Return x, y and the loss function.'''
        x = torch.randn((K,1))
        y = self.slope * x + self.bias
        loss_fct = nn.MSELoss()
        return x, y, loss_fct





@torch.no_grad() # Decorator to disable gradient computation
def sample_task(functional_class = "sinusoid", linear_bias = False, bias_var = 1):
    if functional_class == "sinusoid":
        # for sinusoid tasks
        a = torch.rand(1).item() * 4.9 + .1  # Sample the amplitude in [0.1, 5.0]
        b = torch.rand(1).item() * np.pi  # Sample the phase in [0, pi]
        return Task(a=a, b=b)
    elif functional_class == "linear":
        # for linear tasks
        slope = torch.randn(1).item() + 1
        bias = torch.randn(1) * torch.sqrt(torch.tensor(bias_var)).item() if linear_bias else 0  
        return Task(slope=slope, bias= bias)


def perform_k_training_steps(params, task, batch_size, inner_training_steps, alpha, functional_class='sinusoid', device='cpu'):
    ''' Perform k gradient steps on the task (inner loop)'''
    for epoch in range(inner_training_steps):
        if functional_class == 'sinusoid':
            x_batch, target, loss_fct = task.sample_sinusoid(batch_size)
        elif functional_class == 'linear':
            x_batch, target, loss_fct = task.sample_linear(batch_size)

        loss = loss_fct(mlp(x_batch.to(device), params), target.to(device))

        for p in params:  # Zero grad
            p.grad = None
        gradients = grad(loss, params)
        for p, g in zip(params, gradients):  # Grad step
            p.data -= alpha * g
    return params


def maml(p_model, meta_optimizer, inner_training_steps, nb_epochs, batch_size_K, alpha, nb_tasks=10, functional_class="sinusoid", linear_bias = False, bias_var = 1, device='cpu'):
    """
    Algorithm from https://arxiv.org/pdf/1703.03400v3.pdf (MAML for Few-Shot Supervised Learning)
    """
    training_loss = []
    for epoch in tqdm(range(nb_epochs)):  # Line 2 in the pseudocode

        theta_i_prime = []
        D_i_prime = []

        # Sample batch of tasks
        tasks = [sample_task(functional_class=functional_class, linear_bias=linear_bias, bias_var=bias_var) for _ in range(nb_tasks)]  # Line 3 in the pseudocode
        for task in tasks:
            theta_i_prime.append(
                perform_k_training_steps(
                    params=[p.clone() for p in p_model],
                    task=task,
                    batch_size=batch_size_K,
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
        for i in range(nb_tasks):
            x, y, loss_fct = D_i_prime[i]
            f_theta_prime = theta_i_prime[i]
            # Compute \nabla_theta L(f_theta_i_prime) for task ti
            loss = loss_fct(mlp(x.to(device), f_theta_prime), y.to(device))
            loss.backward()
            batch_training_loss.append(loss.item())

        meta_optimizer.step()  # Line 10 in the pseudocode
        training_loss.append(np.mean(batch_training_loss))
    
    # Save the model
    torch.save({
        'model_state_dict': [p.clone() for p in p_model],
        'meta_optimizer_state_dict': meta_optimizer.state_dict(),
    }, f'models/MAML_M{nb_epochs}_N{nb_tasks}_dim.pth')

    return training_loss


if __name__ == "__main__":
    device = 'cpu'
    functional_class = 'linear'
    linear_bias = False
    bias_var = 1
    params = [torch.rand(40, 1, device=device).uniform_(-np.sqrt(6. / 41), np.sqrt(6. / 41)).requires_grad_(),
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
        batch_size_K=10,
        alpha=1e-3,
        nb_tasks=10,
        functional_class=functional_class,
        linear_bias=linear_bias,
        bias_var=bias_var,
        device=device 
    )

    plt.title('MAML, K=10')
    x = torch.linspace(-5, 5, 50).to(device)
    y = mlp(x[..., None], params)
    plt.plot(x.data.cpu().numpy(), y.data.cpu().numpy(), c='lightgreen', linestyle='--', linewidth=2.2,
             label='pre-update')
    # New task
    task = sample_task(functional_class=functional_class, linear_bias=linear_bias, bias_var=bias_var)
    if functional_class == 'linear':
        ground_truth_y = task.slope * x + task.bias
    elif functional_class == 'sinusoid':
        ground_truth_y = task.a * torch.sin(x + task.b)
    plt.plot(x.data.cpu().numpy(), ground_truth_y.data.cpu().numpy(), c='red', label='ground truth')
    # Fine-tuning, 10 gradient steps
    new_params = perform_k_training_steps(
        params = [p.clone() for p in params], 
        task = task, 
        batch_size=10,
        inner_training_steps=10,
        alpha=1e-3,
        functional_class=functional_class,
        device=device
    )
    # After 10 gradient steps
    y = mlp(x[..., None], new_params)
    plt.plot(x.data.cpu().numpy(), y.data.cpu().numpy(), c='darkgreen', linestyle='--', linewidth=2.2,
             label='10 grad step')
    plt.legend()
    plt.savefig('maml.png')
    plt.show()