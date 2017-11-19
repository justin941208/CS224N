import numpy as np

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def sigmoid_grad(s):
    ds = s * (1 - s)
    return ds

def softmax(x):
    orig_shape = x.shape
    if len(x.shape) > 1:
        x = np.exp(x - x.max(axis=1, keepdims=True))
        x = x / x.sum(axis=1, keepdims=True)
    else:
        x = np.exp(x - x.max())
        x = x / x.sum()
    assert x.shape == orig_shape
    return x

def gradcheck_naive(f, x):
    fx, grad = f(x)
    h = 1e-04
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        H = np.zeros(shape=x.shape)
        H[ix] = h
        numgrad = (f(x+H)[0] - f(x-H)[0]) / (2 * h)

        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        if reldiff > 1e-5:
            print "Gradient check failed."
            print "First gradient error found at index %s" % str(ix)
            print "Your gradient: %f \t Numerical gradient: %f" % (
                grad[ix], numgrad)
            return

        it.iternext()

    print "Gradient check passed!"

def generate_data(N, dimensions):
    X = np.random.randn(N, dimensions[0])
    Y = np.zeros((N, dimensions[-1]))
    for i in xrange(N):
        Y[i, np.random.randint(0,dimensions[-1])] = 1

    Ws = []
    bs = []
    for i in range(1, len(dimensions)):
        Ws.append(np.random.randn(dimensions[i-1], dimensions[i]))
        bs.append(np.random.randn(1,dimensions[i]))
    return X, Y, Ws, bs

def forward_backward_prop(X,Y,W1,b1,W2,b2):
    eta = np.dot(X, W1) + b1
    h = sigmoid(eta)
    Yhat = softmax(np.dot(h, W2) + b2)
    cost = - np.sum(Y * np.log(Yhat))

    #gradb2 = np.sum(Yhat - Y, axis=0, keepdims=True)
    #gradW2 = np.dot(h.T, Yhat - Y)
    #gradb1 = np.sum(np.dot(Yhat - Y, W2.T) * sigmoid_grad(eta), axis=0, keepdims=True)
    gradW1 = np.dot(X.T, np.dot(Yhat - Y, W2.T) * sigmoid_grad(h))

    return cost, gradW1

X,Y,Ws,bs = generate_data(20, [10,5,10])
for i in range(len(Ws)):
    exec('W{} = Ws[i]'.format(i+1))
    exec('b{} = bs[i]'.format(i+1))
gradcheck_naive(lambda W1: forward_backward_prop(X,Y,W1,b1,W2,b2), W1)
