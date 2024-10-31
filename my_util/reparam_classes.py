import torch

from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

class ReparamClass(ABC):
    @abstractmethod
    def ambient_representation(self, x):
        pass
    
    @abstractmethod
    def latent_representation(self, x):
        pass
    
    @abstractmethod
    def derivative(self, x):
        pass

    @abstractmethod
    def get_init_cov_value(self):
        pass
    

class NoReparam(ReparamClass):
    def __init__(self, eps, init_value):
        super().__init__()
        self.__dict__.update(locals())
        
    def ambient_representation(self, x):
        x = torch.clamp(x, min=self.eps, max=None)
        return x

    def latent_representation(self, x):
        return x

    def derivative(self, x):
        return 1.0

    def get_init_cov_value(self):
        return self.init_value

    
class SigmoidReparam(ReparamClass):
    def __init__(self,  a, b, c):
        super().__init__()
        self.__dict__.update(locals())

    def ambient_representation(self, x):
        return ((self.a / (1 + torch.exp(-self.b*x))) + self.c)

    def latent_representation(self, x):
        return (-1/self.b) * torch.log( ((self.a) / (x - self.c)) - 1)

    def derivative(self, x):
        exp_bx = torch.exp( - self.b*x)
        return (   (self.a * self.b * exp_bx) / ((1 + exp_bx)**2)    )
    
    def get_init_cov_value(self):
        return self.ambient_representation(torch.tensor([0]))
    

if __name__ == "__main__":

    choice = "sigmoid"
    if choice == "sigmoid":
        a = 2
        b = 1
        c = 0
        reparam = SigmoidReparam(a, b, c)

    r = 5

    points = torch.arange(-r, r, 0.1)
    x_ambient = reparam.ambient_representation(points.clone())
    x_derivative = reparam.derivative(points.clone())
    
    plt.plot(x_ambient)
    plt.plot(x_derivative)
    plt.show()

