import torch

last_use_cuda = True

def cuda(tensor, use_cuda = None):
    """
    A cuda wrapper
    """
    global last_use_cuda
    if use_cuda == None:
        use_cuda = last_use_cuda
    last_use_cuda = use_cuda
    if not use_cuda:
        return tensor
    if tensor is None:
        return None
    if torch.cuda.is_available():
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
