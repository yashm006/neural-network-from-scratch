import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation 
import functions as f


def function(X:np.array): 
    return np.sin(1/X)

X = np.linspace(-10, 10, 100)
Y = function(X)

class neural_network():
    def __init__(self, X:np.array, Y:np.array):
        # SEED
        np.random.seed(42)
        self.X = X
        self.Y = Y 

        self.x_scale = 1
        self.y_scale = 1


        self.layers = 4    # No. of hidden layers + 1 Output layer
        self.sizes = [1, 128, 64, 42, 1]
        self.eta = 0.01
        self.epochs = 3000

        self.a = [None] * self.layers
        self.z = [None] * self.layers 
        
        # Parameters
        self.W = [None] * self.layers 
        self.b = [None] * self.layers 

        # Gradients
        self.grad_zL = [None] * self.layers
        self.grad_aL = [None] * self.layers
        self.grad_wL = [None] * self.layers
        self.grad_bL = [None] * self.layers

    def normalise(self):
        self.x_scale = np.max(np.abs(self.X))
        self.y_scale = np.max(np.abs(self.Y))
        self.X = self.X / self.x_scale
        self.Y = self.Y / self.y_scale
        print("Normalised automatically.")



    def train_one_epoch(self):
        for i in range(len(self.X)):
            x = self.X[i]
            y = self.Y[i]
            self.forward_pass(x)
            if np.any(np.isnan(self.a[-1])):
                self.normalise()     
                self.init_params()   
                return               
            self.backward_pass(x, y)


    def train(self, epochs = None):
        print("Training Model.. ")
        self.init_params()   
        if (not epochs): epochs = self.epochs


        for epoch in range(epochs):
            for i in range(len(self.X)):
                x = self.X[i] 
                y = self.Y[i] 
                self.forward_pass(x)
                self.backward_pass(x, y)
            print(f"eopch-{epoch} completed")

        print("Model Trained")


    def evaluate(self): 
        L = self.layers - 1
        y_hat = []
        for i in range(len(self.X)): 
            self.forward_pass(self.X[i]) 
            y_hat.append(self.a[L][0])
        return (y_hat, self.Y)
              

    # Initialising the parameters
    def init_params(self):
        L = self.layers - 1
        for i in range(0, L+1):
            self.W[i] = np.random.randn(self.sizes[i+1], self.sizes[i]) * np.sqrt(2 / self.sizes[i])
            self.b[i] = np.zeros((self.sizes[i+1], 1))
    

    # Forward Propagation
    def forward_pass(self, x):
        '''For one pair of $(x,\ y)$'''
        x = np.atleast_1d(x).reshape(-1, 1)   # handles scalar, 0-d array, 1D array
            
        for i in range(self.layers):
            if (i == 0):
                a_1 = x
                self.z[i] = self.W[i] @ a_1 + self.b[i]
                self.a[i] = self.g(self.z[i])
            else:
                self.z[i] = self.W[i] @ self.a[i-1] + self.b[i]
                self.a[i] = self.g(self.z[i])
        self.a[self.layers - 1] = self.f(self.z[self.layers - 1]) 

    
    # Backward Propagation
    def backward_pass(self, x, y):
        '''For one pair of $(x,\ y)$'''
        x = np.atleast_1d(x).reshape(-1, 1)
                                      
        L = self.layers - 1
        y_hat = self.a[L]
        self.grad_zL[L] = (y_hat - y) 
        a_1 = x

        # Step-1: Compute the Gradients
        for i in range(L, -1, -1):
            if (i == L): 
                self.grad_wL[i] = self.grad_zL[i] @ self.a[i-1].T
                self.grad_bL[i] = self.grad_zL[i]

            if (i == 0):
                self.grad_aL[i] = self.W[i+1].T @ self.grad_zL[i+1]
                self.grad_zL[i] = self.grad_aL[i] * self.g_prime(self.z[i])
                self.grad_wL[i] = self.grad_zL[i] @ x.T
                self.grad_bL[i] = self.grad_zL[i]                

            elif (i < L):
                self.grad_aL[i] = self.W[i+1].T @ self.grad_zL[i+1]
                self.grad_zL[i] = self.grad_aL[i] * self.g_prime(self.z[i])
        
                self.grad_wL[i] = self.grad_zL[i] @ self.a[i-1].T
                self.grad_bL[i] = self.grad_zL[i]

        
        # Step-2: Update the Weights and Bias
        for i in range(0, L+1):
            self.W[i] = self.W[i] - self.eta*self.grad_wL[i]
            self.b[i] = self.b[i] - self.eta*self.grad_bL[i]
    
            

    def g(self, z):
        return np.where(z > 0, z, 0.01 * z)

    def g_prime(self, z):
        return np.where(z > 0, 1.0, 0.01)


    def f(self, z):
        return z



# ── Setup ─────────────────────────────────────────────────────────
n = neural_network(X, Y)
n.init_params()

plt.ion()   # interactive mode ON
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(X, Y, 'k--', lw=1.5, label='Target f(x)')
pred_line, = ax.plot(X, np.zeros_like(X), 'r-', lw=2, label='NN estimate')
ax.set_xlim(X.min() - 0.25*X.max(), X.max() + 0.25*X.max())
ax.set_ylim(Y.min() - 0.25*Y.max(), Y.max() + 0.25*Y.max())
ax.legend()
title = ax.set_title('Epoch 0')


# ── Train + Plot live ─────────────────────────────────────────────
X_plot = X  # sorted, for plotting only

for epoch in range(n.epochs):
    # reshuffle whatever is currently in n.X, n.Y (could be normalized)
    idx = np.random.permutation(len(n.X))
    n.X, n.Y = n.X[idx], n.Y[idx]

    n.train_one_epoch()
    
    y_hat = []
    for xi in X / n.x_scale:    
        n.forward_pass(xi)
        y_hat.append(n.a[n.layers - 1][0] * n.y_scale)
    
    pred_line.set_ydata(np.array(y_hat).flatten())
    title.set_text(f'Epoch {epoch + 1} / {n.epochs}')
    plt.pause(0.01)


plt.ioff()
plt.show()