from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        self.batchnorm = {} # 增加这一行，将beta、gamma存入，存入params会与slover冲突
        self.layernorm = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # pass
        # 维度数组进行拼接，hstack将元素按水平方向叠加
        # input_dim =15 ,hidden_dims = [20,30] ,num_classes =10, layer_dims =[15,20,30,10]
        layer_dims = np.hstack((input_dim, hidden_dims, num_classes))

        # 初始化W和b,W服从正态分布标准差0，b初始为0
        for i in range(self.num_layers):
          W = np.random.normal(loc=0.0, scale=weight_scale, size=(layer_dims[i], layer_dims[i+1]))
          b = np.zeros(layer_dims[i+1])
          self.params['W'+str(i+1)] = W
          self.params['b'+str(i+1)] = b
        # 使用批归一化，对每层存储一个beta和gamma
        if normalization == 'batchnorm':
          for i in range(self.num_layers-1):
            gamma = np.ones(layer_dims[i+1])
            beta = np.zeros(layer_dims[i+1])
            self.batchnorm['gamma'+str(i+1)] = gamma
            self.batchnorm['beta'+str(i+1)] = beta
        elif normalization == 'layernorm':
          for i in range(self.num_layers-1):
            gamma = np.ones(layer_dims[i+1])
            beta = np.zeros(layer_dims[i+1])
            self.layernorm['gamma'+str(i+1)] = gamma
            self.layernorm['beta'+str(i+1)] = beta


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.
        
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # pass
        x=X
        caches=[]   # 保存有网络的中间信息
        caches_bn=[]
        caches_ln=[]
        caches_drop=[]
        for i in range(self.num_layers-1):
          W=self.params['W'+str(i+1)]
          b=self.params['b'+str(i+1)]

          # 无 BN，无 dropout：affine-layernorm-relu-dropout
          if self.normalization == None:
            out,cache = affine_relu_forward(x,W,b)
          elif self.normalization == 'batchnorm':
            out,cache = affine_forward(x,W,b) # affine-batchnorm-relu-droupout
            out,cache_bn = batchnorm_relu_forward(out,self.batchnorm['gamma'+str(i+1)],self.batchnorm['beta'+str(i+1)],
              self.bn_params[i]) # do not forget gamma & beta
            caches_bn.append(cache_bn)
          elif self.normalization == 'layernorm':
            out,cache = affine_forward(x,W,b)
            out,cache_ln = layernorm_relu_forward(out,self.layernorm['gamma'+str(i+1)],self.layernorm['beta'+str(i+1)],
              self.bn_params[i])
            caches_ln.append(cache_ln)
          # 保存
          if self.use_dropout:
            out,cache_drop = dropout_forward(out, self.dropout_param)
            caches_drop.append(cache_drop)
          caches.append(cache)# 保存了当前层的输入信息
          x=out
        # 最后一层，affine
        scores,cache = affine_forward(x,self.params['W'+str(self.num_layers)],self.params['b'+str(self.num_layers)])
        caches.append(cache)    # 保存了最后一层的输入信息，为(x,w,b)


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # pass
        # 计算损失和梯度，添加L2正则化强度
        loss,dscores = softmax_loss(scores,y)	# 计算损失（不完整）以及反向传播过来的梯度
        for i in range(self.num_layers):	# 一定要记得加上L2正则惩罚项，这样就计算得到完整的损失了
          W = self.params['W'+str(i+1)]
          loss += 0.5 * self.reg * np.sum(W**2)

        # 计算最后一层传播过来的梯度,即隐藏层的最后一层，输出层没有w和b，dscores为输出层的梯度
        dout,dW,db = affine_backward(dscores,caches[self.num_layers-1])
        dW += self.reg * self.params['W'+str(self.num_layers)]

        grads['W'+str(self.num_layers)]=dW
        grads['b'+str(self.num_layers)]=db

        # 计算前n-1层的梯度
        for i in range(self.num_layers-2,-1,-1):
          if self.use_dropout:
            dout = dropout_backward(dout, caches_drop[i])
          if self.normalization == None:
            dout,dW,db = affine_relu_backward(dout,caches[i])
            dW += self.reg * self.params['W'+str(i+1)]
            grads['W'+str(i+1)]=dW
            grads['b'+str(i+1)]=db
          elif self.normalization == 'batchnorm':
            dout,dW,db = batchnorm_relu_backward(dout,caches_bn[i])
            dout,dW,db = affine_backward(dout,caches[i])
            dW += self.reg * self.params['W'+str(i+1)]
            grads['W'+str(i+1)]=dW
            grads['b'+str(i+1)]=db
          elif self.normalization == 'layernorm':
            dout,dW,db = layernorm_relu_backward(dout,caches_ln[i])
            dout,dW,db = affine_backward(dout,caches[i])
            dW += self.reg * self.params['W'+str(i+1)]
            grads['W'+str(i+1)] = dW
            grads['b'+str(i+1)] = db


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
