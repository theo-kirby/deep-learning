from tensor import Tensor # symlink from tensor-math
from activations import sigmoid, de_sigmoid

class CNN():

    def __init__(self):
        pass

    def convolve(self, x: Tensor, k: Tensor, bias: float = 0.0) -> Tensor:

        # x is a 2d matrix with a width and height
        # k is the kernel tensor, a patch shape that slides over the input matrix

        assert x.dim == 2
        assert k.dim == 2

        height, width = x.shape # height and width of the input matrix
        k_height, k_width = k.shape # height and width of the kernel patch (filter)

        out_height = height - k_height + 1
        out_width = width - k_width + 1

        assert out_height > 0 and out_width > 0

        # create the output tensor (width x height of how many full patches fit in the input matrix) *cnn.0.3
        output = Tensor((out_height, out_width))

        # iterate through the empty output matrix and calculate the convolution for each position
        # basically take some x,y (every x,y of input) and place the kernel on it, then iterate through each cell of the kernel size and
        # set the value of the output cell equal to the dot product of the input patch and the kernel
        for out_y in range(out_height): # row of out
            for out_x in range(out_width): # column of out
                s = 0.0
                for ky in range(k_height): # row of k
                    for kx in range(k_width): # column of k
                        s += x[(out_y + ky, out_x + kx)] * k[(ky, kx)]
                output[(out_y, out_x)] = s + bias

        return output