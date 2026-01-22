import math

def ELU(x,n,a=0.01):
    if x > n:
        return x
    elif x <= n:
        return torch.tensor([a*(math.exp(x)-1)])
