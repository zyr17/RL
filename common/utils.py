import torch

ENABLE_CUDA = True

def cuda(tensor):
    """
    A cuda wrapper
    """
    if tensor is None:
        return None
    if torch.cuda.is_available() and ENABLE_CUDA:
        return tensor.cuda()
    else:
        return tensor

class Fake_TXSW:
    def __init__(self):
        pass
    def add_scalar(self, *x):
        pass
    def add_image(self, *x):
        pass
    def add_graph(self, *x):
        pass
    def close(self):
        pass

def showarray(arr):
    arr = np.array(arr)
    print('max: %.2f, min: %.2f' % (arr.max(), arr.min()))
    plt.imshow(arr)
    plt.show()
