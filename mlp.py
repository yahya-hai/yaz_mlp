import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 

def gram_schmidt(X: np.array) -> np.array:
    """  
    Creates an orthonormal basis of a matrix  
    """
    zero = np.zeros((X.shape[0], 1)) #threshold instead?

    x1 = X[:, 0].reshape((-1, 1))
    norm = np.linalg.norm(x1)
    u1 = x1/norm
    C = u1

    for k in range(1, X.shape[1]):
        x = X[:, k].reshape((-1, 1))
        
        x_prime = x - (np.dot(np.dot(C, C.T), x))
        if not np.all(x_prime == zero):
            norm = np.linalg.norm(x_prime)
            u = x_prime/norm
            C = np.hstack((C, u))
    U = C
    return U

class layer:

    def __init__(self, dims: tuple[int, int]):
        self.rows = dims[0]
        self.cols = dims[1]
        self.W = np.random.randn(self.cols, self.rows) 
        self.b = np.random.randn(self.cols, 1) 
        self.A = np.ones((self.cols, 1))
        self.Z = np.ones((self.cols, 1))
        self.Z_Prime = np.ones((self.cols, 1))
        self.X = np.ones((self.cols, 1))

    def activation(self, x: float):
        pass

    def activation_deriv(self, x: float):
        pass

    def calc_z(self, x: np.array):
        self.Z = np.dot(self.W, x) + self.b
        return self.Z
    
    def activation_deriv_vectorized(self):
        vectorized_activation_deriv = np.vectorize(self.activation_deriv)
        self.Z_Prime = vectorized_activation_deriv(self.Z)
        return self.Z_Prime

class sigmoid_layer(layer):

    def __init__(self, dims: tuple[int, int]):
        super().__init__(dims)

    def activation(self, x: float):
        return 1/(1 + np.exp(-x))

    def activation_vectorized(self):
        #activation of the layer; uses the net input attribute
        vec_act = np.vectorize(self.activation)
        self.A = vec_act(self.Z)

    def activation_deriv(self, x: float):
        sgm_prime = self.activation(x)
        return sgm_prime * (1 - sgm_prime)
    

class tanh_layer(layer):
    def __init__(self, dims: tuple[int, int]):
        super().__init__(dims)

    def activation(self, x: float):
        return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

    def activation_vectorized(self):
        #activation of the layer; uses the net input attribute
        vec_act = np.vectorize(self.activation)
        self.A = vec_act(self.Z)

    def activation_deriv(self, x: float):
        activ = self.activation(x)
        return 1 - (activ ** 2)
    
class relu_layer(layer):
    
    def __init__(self, dims: tuple[int, int]):
        super().__init__(dims)

    def activation(self, x: float):
        return max(x, 0)
    
    def activation_vectorized(self):
        vec_act = np.vectorize(self.activation)
        self.A = vec_act(self.Z)

    def activation_deriv(self, x: float):
        return x > 0

class softmax_layer(layer):
    
    def __init__(self, dims: tuple[int, int]):
        super().__init__(dims)
        self.sum = 0
    
    def activation(self, x: float):
        pass 

    def activation_vectorized(self):
        denom = np.sum([np.exp(z) for z in self.Z])
        self.sum = denom

        res = np.array([np.exp(num)/denom for num in self.Z])
        res = res.reshape((-1, 1))
        self.A = res

    def activation_deriv(self, x: float):
        pass
    
    def activation_deriv_vectorized(self):
        denom = self.sum ** 2
        res = np.array([((self.sum * np.exp(z)) - (np.exp(z) ** 2))/denom for z in self.Z])
        res = res.reshape((-1, 1))
        self.Z_Prime = res 
        return self.Z_Prime

class network():

    def __init__(self, layers: list[int], dark_mode: int=0):
        self.dark_mode = dark_mode
        pass

    def switch_modes(self, new_mode: int) -> None:
        #turns dark mode on and off
        assert(new_mode in [0, 1])
        self.dark_mode = new_mode

    def add_relu_layer(self, layer: tuple[int, int]):
        self.layers.append(relu_layer(dims=layer))
         
    def add_sigmoid_layer(self, layer: tuple[int, int]):
        self.layers.append(sigmoid_layer(dims=layer))
         
    def add_softmax_layer(self, layer: tuple[int, int]):
        self.layers.append(softmax_layer(dims=layer))

    def feedforward(self, X: np.array):
        #make sure x is an nx1 vector
        #first feedforward for the first one, then loop over the rest 
        first = self.layers[0]
        
        first.calc_z(X)
        first.activation_vectorized()
        first.X = X
        
        for i in range(1, len(self.layers)):
            prev_layer = self.layers[i - 1]
            cur_layer = self.layers[i]
            
            x = prev_layer.A
            cur_layer.X = x
            cur_layer.calc_z(x)
            cur_layer.activation_vectorized()

        return self.layers[-1].A

    def error(self, y: np.array, type: str='MSE'):
        #calculates error
        n = max(y.shape)
        if type == 'MSE':
            y_hat = self.layers[-1].A
            return ((np.linalg.norm(y - y_hat)) ** 2) * (1/n)

    def error_deriv(self, y: np.array, type: str='MSE'):
        #Fixed it!
        n = max(y.shape)
        if type == 'MSE':
            y_hat = self.layers[-1].A
            return  (2/n) * (y_hat - y)

    def backprop_single(self, y: np.array):
        """ 
        res = list[tuple[np.array, np.array]]
        """
        res = []

        #do the work for the outer layer, then the rest
        outer_layer = self.layers[-1]
        y = outer_layer.A

        dEdA = self.error_deriv(y)
        dAdZ = outer_layer.activation_deriv_vectorized() 
        dZdW = outer_layer.X


        delta = np.multiply(dEdA, dAdZ)

        dEdW = np.dot(delta, dZdW.T)
        dEdb = delta

        res.insert(0, (dEdW, dEdb))

        n = len(self.layers) - 2 
        
        while n >= 0:
            cur_layer = self.layers[n]
            layer_ahead = self.layers[n+1]
            
            delta_new = delta
            dZdA = layer_ahead.W 
            dAdZ = cur_layer.activation_deriv_vectorized() 
            dZdW = cur_layer.X 

            term = np.dot(dZdA.T, delta)
            delta_new = np.multiply(term, dAdZ)

            dEdW = np.dot(delta_new, dZdW.T)
            
            dEdb = delta_new
            delta = delta_new

            res.insert(0, (dEdW, dEdb))
            n -= 1
        return res
    
    def sum_nested_gradients(self, data):
        length = len(data[0])
        batch_size = len(data)
        
        result = []
        for i in range(length):
            elems = [tup[i] for tup in data]
            arrays1, arrays2 = zip(*elems)
            sum1 = sum(arrays1)
            sum2 = sum(arrays2)
            result.append((sum1, sum2))

        for tup in result:
            new_tup = (tup[0]/batch_size, tup[1]/batch_size)
            tup = new_tup
        return result
        
    def get_data(self, data: pd.DataFrame) -> tuple[np.array, np.array]:
        # fetches some data
        max = len(data.index)
        row = np.random.randint(0, max)
        x = data['features'][row]
        y = data['labels'][row]
        return x, y
    
    def apply_grads(self, grads: list[tuple], lr: float=0.001):
        # helper for subtracting gradients from parameters
        for idx, pair in enumerate(grads):
            W, b = pair 
            self.layers[idx].W -= (lr * W)
            self.layers[idx].b -= (lr * b)        

    def train(self, data: pd.DataFrame, batch_size: int=16, 
            learning_rate: float=0.001, err_type: str='MSE'): 
        #Training and updating for a single epoch
        shuffled = data.sample(frac=1).reset_index(drop=True)
        errors = []
        max = len(shuffled)
        for i in range(0, max, batch_size):
            batch = shuffled[i:i+batch_size]
            derivs = []
            err = 0
            for _, row in batch.iterrows():
                x, y = row['features'], row['labels']
                self.feedforward(x)
                deriv = self.backprop_single(y) 
                derivs.append(deriv)
                error = self.error(y, err_type)
                err += np.linalg.norm(error)
            final_derivs = self.sum_nested_gradients(derivs)
            errors.append(err)
            self.apply_grads(final_derivs, learning_rate)
        final_error = np.sum(errors, axis=0)
        return final_error

    def test(self, eval: pd.DataFrame):
        #tests the accuracy of the neural network on some training data
        max = len(eval.index)
        correct = 0
        for row in range(max):
            x = eval['features'][row]
            y = eval['labels'][row]
            y_hat = self.feedforward(x)
            ans = np.argmax(y)
            guess = np.argmax(y_hat)
            if guess == ans:
                correct += 1
        return (correct/max) * 100

    def train_epochs(self, train: pd.DataFrame, 
        test: pd.DataFrame, epochs: int=1000, 
        batch_size: int = 16, learning_rate: float = 0.001,
        err_type: str='MSE'):
        #training loop
        x_axis_1 = [] #epoch number
        x_axis_2 = [] 
        ers = []
        gambling = []

        for i in range(1, epochs+1):
            x_axis_1.append(i)
            res = self.train(train, batch_size, learning_rate, err_type)
            ers.append(res)
            if i % 10 == 0 or i == 1: # every 10 epochs, we're going to test
                guesses = self.test(test)
                x_axis_2.append(i)
                gambling.append(guesses)

        #dark mode for night owls
        if self.dark_mode:
            plt.style.use("dark_background")
        

        plt.subplot(2, 1, 1)
        plt.plot(x_axis_1, ers, color='blue')
        plt.xlabel("Epoch #")
        plt.ylabel("Error")
        plt.title(("Model Performance: Batch Size = {}".format(batch_size)))

        plt.subplot(2, 1, 2)
        plt.scatter(x_axis_2, gambling, color='red')
        plt.xlabel("Epoch #")
        plt.ylabel("% Correct Guesses")
        plt.title(("Model Performance: Batch Size = {}").format(batch_size))

        plt.tight_layout(pad=2.5)
        plt.show()
        
class sigmoid_network(network):
    def __init__(self, layers: list[int], dark_mode: int=0):
        self.dark_mode = dark_mode
        self.layers = [sigmoid_layer((layers[i], layers[i+1])) for i in range(len(layers) - 1)]

class relu_network(network):
    def __init__(self, layers: list[int], dark_mode: int=0):
        self.dark_mode = dark_mode
        self.layers = [relu_layer((layers[i], layers[i+1])) for i in range(len(layers) - 1)]

class tanh_network(network):
    def __init__(self, layers: list[int], dark_mode: int=0):
        self.dark_mode = dark_mode
        self.layers = [tanh_layer((layers[i], layers[i+1])) for i in range(len(layers) - 1)]

class softmax_network(network):
    #who would ever want this???
    def __init__(self, layers: list[int], dark_mode: int=0):
        self.dark_mode = dark_mode
        self.layers = [softmax_layer((layers[i], layers[i+1])) for i in range(len(layers) - 1)]
