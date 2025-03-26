import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.autograd.functional import jacobian, jvp


class MyJacobianRegularizer(nn.Module):
    def __init__(self, n_proj=1):
        super(MyJacobianRegularizer, self).__init__()
        assert n_proj == -1 or n_proj > 0
        self.n_proj = n_proj

    def forward(self, x, y, y_star=None):
        # y = f(x); we want to penalize ||dy/dx||_F^2, or if we want the Jacobians to match some model,
        # we penalize ||dy/dx - J^*||_F^2, where J^* is the Jacobian of the model
        B, nx = x.shape
        _, ny = y.shape

        assert self.n_proj <= ny, "n_proj must be less than or equal to the output dimension"

        vs = self._random_unit_vector(B, ny, device=y.device)

    def _jvp_single(self, x, y, v):
        _, out = jvp(y, (x, ), (v, ))

    def _random_unit_vector(self, B, C, device):
        v = torch.randn(B, C, device=device)
        return v / torch.norm(v, dim=1, keepdim=True)


class JacobianReg(nn.Module):
    '''
    Loss criterion that computes the trace of the square of the Jacobian.

    Arguments:
        n (int, optional): determines the number of random projections.
            If n=-1, then it is set to the dimension of the output 
            space and projection is non-random and orthonormal, yielding 
            the exact result.  For any reasonable batch size, the default 
            (n=1) should be sufficient.
    '''
    def __init__(self, n=1):
        assert n == -1 or n > 0
        self.n = n
        super(JacobianReg, self).__init__()

    def forward(self, x, y):
        '''
        computes (1/2) tr |dy/dx|^2
        '''
        B, C = y.shape
        if self.n == -1:
            num_proj = C
        else:
            num_proj = self.n
        J2 = 0
        for ii in range(num_proj):
            if self.n == -1:
                # orthonormal vector, sequentially spanned
                v = torch.zeros(B, C)
                v[:, ii] = 1
            else:
                # random properly-normalized vector for each sample
                v = self._random_vector(C=C, B=B)
            if x.is_cuda:
                v = v.cuda()
            Jv = self._jacobian_vector_product(y, x, v, create_graph=True)
            J2 += C * torch.norm(Jv) ** 2 / (num_proj * B)
        R = (1 / 2) * J2
        return R

    def _random_vector(self, C, B):
        '''
        creates a random vector of dimension C with a norm of C^(1/2)
        (as needed for the projection formula to work)
        '''
        if C == 1:
            return torch.ones(B)
        v = torch.randn(B, C)
        arxilirary_zero = torch.zeros(B, C)
        vnorm = torch.norm(v, 2, 1, True)
        v = torch.addcdiv(arxilirary_zero, 1.0, v, vnorm)
        return v

    def _jacobian_vector_product(self, y, x, v, create_graph=False):
        '''
        Produce jacobian-vector product dy/dx dot v.

        Note that if you want to differentiate it,
        you need to make create_graph=True
        '''
        flat_y = y.reshape(-1)
        flat_v = v.reshape(-1)
        grad_x, = torch.autograd.grad(flat_y, x, flat_v, retain_graph=True, create_graph=create_graph)
        return grad_x
