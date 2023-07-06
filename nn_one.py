import random
import numpy as np

def sigmoid(x):
    return ((1)/(1+np.exp(-x)))

#d/dx of (1/1+e^-x) = (e^-x)/(1+e^-x)^2 which is sigmoid*(1-sigmoid)
def sigmoid_prime(x):
    temp = sigmoid(x)
    return (temp*(1-temp))

def cost_derivative(a_l,y):
    return (a_l-y)

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        #For each layer, you would need a bias for each neuron in it, thus you'd need size number of biases
        #However, 1st entry of bias corresponds to second layer(not 1st or input layer). Thus n-1th entry will correspond
        #to output layer or last layer.
        self.weights = [np.random.randn(y, x) 
                        for x, y in zip(sizes[:-1], sizes[1:])]
        #Again from second to last layer, but each neuron's number of weights/values to store would be the size of previous layer(x).
        #zip basically creates pairs/tuples.
        #example zip([1,2,3,4,5], [1,4,9,16,25] would give (1,1),(2,4),(3,9),(4,16),(5,25))

#In_python3, division is float by default. Therefore, you don't need to explicitly write 1.0

    def get_last_layer(self, a):
        for b,w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a)+b)
        return a

#input format of training_data: x,y such that x is input and y is output
    def SGD(self, n_epochs, training_data, mini_batch_size, eta, test_data = None):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        #xrange in python2 is equivalent to the range in python3
        for i in range(n_epochs):
            random.shuffle(training_data)
            #cut the training data into mini batches of given size(parameter m_b_s) and store it in mini_batches
            #####mini_batches = []
            #####for k in range(0,n,mini_batch_size):
                #####mini_batches += [training_data[k:k+mini_batch_size]]
            #the above code doesn't work because it returns the same array as training data, ie, no partitions(although we add elements by partitions)
            #####mini_batches = [training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
            #the above code works,actually nielsen's code, but i wanted to try something different
            mini_batches = []
            for k in range(0,n,mini_batch_size):
                mini_batches.append(training_data[k:k+mini_batch_size])
            #count =0  
            #therefore each mini batch has a subset of training data i.e (x,y), x: input y: output
            for mini_batch in mini_batches:
                #now calculate the path of descent using gradient and update the weights and biases accordingly
                #the updation happens within the function call
                self.update_weights_and_biases(mini_batch, eta)
                '''print("MINIBATCH COUNT : ", count)
                count+=1'''  
            if test_data:
                #printing size of test data just for readability
                print("Epoch {0} is over. {1}/ {2} correct".format(i,self.evaluate(test_data), n_test))
            else:
                print(" The {0}th epoch has been completed" .format(i))

    def update_weights_and_biases(self, mini_batch, eta):
        #basic idea is to take each tuple x,y and using backprop calculating derivative to weights and biases
        #in the end you sum all these partial derivatives and average it(divide by size of mini batch)
        #We are taking steps at 'eta' times the derivative. Hence eta aka "Learning Rate"
        #Naming/Notation followed is from 3b1b neural network playlist(4th video)
        delC0_by_delw = [np.zeros(layer.shape) for layer in self.weights]
        delC0_by_delb = [np.zeros(layer.shape) for layer in self.biases]
        for x,y in mini_batch:
            #find_wish_of_point tells us how that particular point(input and desired output) wants to modify the weights and biases(-ve)
            #if that point had ultimate power
            xy_delC0_by_delw, xy_delC0_by_delb = self.find_wish_of_point(x,y)
            '''print("Size of w gradient : ", xy_delC0_by_delw.size)
            print("Size of b gradient : ", xy_delC0_by_delb.size)
            print("Size of w  : ", delC0_by_delw.size)
            print("Size of b : ", delC0_by_delb.size)'''
            delC0_by_delw = [total+point for total, point in zip(delC0_by_delw,xy_delC0_by_delw)]
            delC0_by_delb = [total+point for total, point in zip(delC0_by_delb,xy_delC0_by_delb)]
        #now that we are done with the mini batch, time to update weights and biases as promised
        self.weights = [w-(step_w/len(mini_batch))*eta for w,step_w in zip(self.weights, delC0_by_delw)]
        self.biases = [b-(step_b/len(mini_batch))*eta for b,step_b in zip(self.biases, delC0_by_delb)]
        #note that we are dividing by size of mini batch(it is prolly just a matter of taste, not necessary) coz we are taking average
        
    def find_wish_of_point(self,x,y):
        #Finding how the point wishes to change the weights and biases so as to minimize cost
        #(basically finding gradient at that particular instant/point)
        xy_delC0_by_delw = [np.zeros(layer.shape) for layer in self.weights]
        xy_delC0_by_delb = [np.zeros(layer.shape) for layer in self.biases]
        a = x #x is the input layer of the NN, activation is a temp variable used to process x on the NN
        #print(a.shape)
        a_list = [x] #a list which will keep track of all the activations of the NN on input x
        z_list = []
            #Now we will begin simulating x on the network to see what output we get
        for b,w in zip(self.biases, self.weights):
            '''print(b.shape, " is the shape of bias")
            print(w.shape, " is the shape of weight")
            print(a.shape, " is the shape of activation")'''
            z = np.dot(w,a) + b
            z_list.append(z)
            a = sigmoid(z)
            a_list.append(a)
        delta = cost_derivative(a_list[-1],y) * sigmoid_prime(z_list[-1])
        #print(delta.shape, " is the shape of delta")
        xy_delC0_by_delb[-1] = delta
        xy_delC0_by_delw[-1] = np.dot(delta,a_list[-2].transpose())
        #print(xy_delC0_by_delw[-1].shape, " is the shape of last layer weight grad")
        #let ri denote reverse index
        for ri in range(2,self.num_layers):
            #I've understood how this works in the simple 1 neuron per layer case. For the multiple neurons
                #I have taken an example and seen that this works, but I couldn't derive the matrix multi. from scratch
                #Written it down in a paper 
            z = z_list[-ri] 
            zp = sigmoid_prime(z)
            delta = np.dot(self.weights[-ri+1].transpose(), delta) * zp
            xy_delC0_by_delb[-ri] = delta
            xy_delC0_by_delw[-ri] = np.dot(delta,(a_list[-ri-1]).transpose())
            #write_to_avoid_error = True
        return(xy_delC0_by_delw,xy_delC0_by_delb)
    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.get_last_layer(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
