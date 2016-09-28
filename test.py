
import numpy as np
import matplotlib.pyplot as plt


def load_train_data():
    """Loads the training data from file"""
    train_file = np.load('auTrain.npz')
    images = train_file['digits']
    labels = train_file['labels']
    #print('Shape of input data: %s %s' % (images.shape, labels.shape))
    return images, labels

def load_test_data():
    test_file = np.load('auTest.npz')
    #print(test_file.keys())
    images_test = test_file['digits']
    labels_test = test_file['labels']
    #print('Shape of test input data: %s' % (images_test.shape,))
    return images_test, labels_test

def logistic(z):
    """ 
    Computes the logistic function to each entry in input vector z.

    Args:
        z: A vector (numpy array) 
    Returns:
        A new vector of the same length where each entry is the value of the 
        corresponding input entry after applying the logistic function to it
    """
    
    # Calculate the logistic function
    result = np.exp(-np.logaddexp(0, -z))
    # The result should be greater than 0 and less than 1, so we set all values outisde of
    # interval to a proper representation of their value.
    # This is to fix issues of floats being slightly incorrect.
    result[result <= 0] = np.nextafter(0,1)
    result[result >= 1] = np.nextafter(1,0)
    return result

def log_cost(X, y, w=None, reg=0):
    """
    Compute the regularized cross entropy and the gradient under the logistic regression model 
    using data X,y and weight vector w

    Args:
        X: A 2D numpy array of data points stored row-wise (n x d)
        y: A 1D numpy array of target values (array of length n)
        w: A 1D numpy array of weights (d x 1)
        reg: Optional regularization parameter
    Returns:
        cost: Average Negative Log Likelihood of w
        grad: The gradient of the average Negative Log Likelihood at w
    """
    if w is None:
        w = np.zeros((X.shape[1], 1))
    
    # Vectorizing logistic.
    # shape the targetvalues as a one-dimensional matrix
    n, d = X.shape
    y = y.reshape(n, 1)
    
    # calculate the logistic function that will be used repeatedly
    sigma = logistic(X.dot(w))
    # Avoid using sigma's == 1 in NLL calculation (avoid log(0)'s)
    cost = (- np.sum( y*np.log(sigma) + (1 - y)*np.log(1-sigma)) + 0.5*reg*np.linalg.norm(w, ord=2))/n
    grad = (- X.T.dot(y-logistic(X.dot(w))) + reg*np.linalg.norm(w, ord=2))/n
    return cost, grad

def batch_grad_descent(X, y, w=None, reg=0, stepsize=0.1, maxiter=2**12,):
    """ 
    Run Batch Gradient Descent on data X,y to minimize the NLL for logistic regression on data X,y

    Args:
        X: A 2D numpy array of data points stored row-wise (n x d)
        y: A 1D numpy array of target values (n x 1)
        w: A 1D numpy array of weights (d x 1)
        reg: Optional regularization parameter
    Returns:
        Learned weight vector (d x 1)
    """
    if w is None:
        w = np.zeros((X.shape[1], 1))
        
    bestw = w
    bestcost, grad = log_cost(X, y, w)
    bestcost = bestcost*10
    for itercount in range(0, maxiter):
        cost, grad = log_cost(X, y, w, reg=reg)
        w = w - stepsize*grad
        stepsize = max(0.00001, 1.0 - (itercount/maxiter))
        if cost < bestcost:
            bestcost = cost
            bestw = w
    return bestw, bestcost

def mini_batch_grad_descent(X, y, w=None, reg=0, batchsize=100, epochs=2*12, stepsize=0.01):
    """
    Run Mini-Batch Gradient Descent on data X,y to minimize the NLL for logistic regression on data X,y
    The input defines mini-batch size and the number of passess through the data set (epochs)

    Args:
        X: A 2D numpy array of data points stored row-wise (n x d)
        y: A 1D numpy array of target values (n x 1)
        w: A 1D numpy array of weights (d x 1)
        reg: Optional regularization parameter
        batchsize: Number of data points to use in each batch
        epochs: Number of times to go over all data points
    Returns:
        Learned weight vector (d x 1)
    """
    if w is None:
        w = np.zeros((X.shape[1], 1))
    
    # Please implement me and use "batchsize" data points in each gradient computation!
    n = X.shape[0]
    #if n % batchsize != 0:
        #print('Some datapoints are discarded every epoch to achieve the specified batchsize.')
        #print(n)
    bestw = w
    bestcost, grad = log_cost(X, y, w)
    epochcount = 0
    for epoch in range(0,epochs):
        w = bestw
        idx = np.random.permutation(n)
        for batch in range(0,n//batchsize):
            # a batchsize-slice of the randomly permutation and run grad_descent (only one iteration!) 
            idxb = idx[batch*batchsize:(batch+1)*batchsize]
            w, cost = batch_grad_descent(X[idxb,:], y[idxb], w, reg=reg, stepsize=stepsize, maxiter=1,)
            #w, cost = batch_grad_descent(X, y, w, reg=reg, stepsize=stepsize, maxiter=2**8,)
        if cost < bestcost:
            bestcost = cost
            bestw = w
        stepsize = max(0.00001, 1.0 - (epoch/epochs))
        #print('          NLL: ', cost, 'after ', epoch+1, ' epochs.')
    return bestw, bestcost

def twosAndSevens(reg=0):
    #print('You rang sir/miss')
    digits, labels = load_train_data()
    idx1 = (labels == 2)
    idx2 = (labels == 7)
    img27 = digits[idx1 | idx2,:]
    lab27 = labels[idx1 | idx2]
    lab27[lab27==2] = 0
    lab27[lab27==7] = 1
    ## code for plotting and checking the logistic()-function
    #x = np.arange(-20,20,0.5)
    #plt.scatter(x,logistic(x))
    #plt.axis([-10, 10, -0.1, 1.1])
    
    ## How does the all-zero vector perform:
    
    #cost, grad = log_cost(img27, lab27)
    #w = np.ones((img27.shape[1], 1))
    #print('    Start NLL: ', cost)
    
    ## Try the batch_grad_descent
    #print('batch_grad_descent')
    #w, cost = batch_grad_descent(img27, lab27)
    #print('    Final NLL: ', cost)
    
    ## Try the mini_batch_grad_descent
    print('mini_batch_grad_descent')
    w, cost = mini_batch_grad_descent(img27, lab27, reg=reg)
    print('    Final NLL: ', cost)
    
    
    plt.imshow(w.reshape(28, 28), cmap='bone')
    print('Look look it is a pretty two i think')
    
    # Now we should test our data.
    digits_test, labels_test = load_test_data()
    idx1 = (labels_test == 2)
    idx2 = (labels_test == 7)
    img27 = digits_test[idx1 | idx2,:]
    lab27 = labels_test[idx1 | idx2]
    lab27[lab27==2] = 0
    lab27[lab27==7] = 1
    lab27
    
    Y = np.dot(img27, w)
    Y[Y >= 0] = 1
    Y[Y < 0] = 0
    n, d = Y.shape
    lab27 = lab27.reshape(n, 1)
    
    print("Number of wrongly classified")
    result = np.squeeze(np.asarray((Y != lab27)))
    wrong = result.sum()
    print(str(wrong) + " : " + "{0:.2f}%".format(wrong/result.shape[0] * 100))
    
    wrongClass = img27[result]
    print("Theese i could not classify")
    
    #for i in range(wrongClass.shape[0]):
    #    plt.figure(i+1)
    #    plt.imshow(wrongClass[i].reshape(28, 28), cmap='bone')
    #return result.sum()
    
if __name__ == '__main__':
    twosAndSevens()
    