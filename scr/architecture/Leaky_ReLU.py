
def leaky_ReLU(x, n, a=0.01):
    if x >= n:
        return x
    elif x < n:
        return x*a

