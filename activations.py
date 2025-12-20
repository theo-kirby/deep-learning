from math import exp # exp (e)

# TODO: add tanh, relu
def sigmoid(x: float) -> float:
    if x >= 0:
        return 1 / (1 + exp(-x))
    else:
        # save overflow if neg
        ex = exp(x)
        return ex / (1 + ex)

def de_sigmoid(x: float) -> float:
    return x * (1 - x) # expects already sigmoided x