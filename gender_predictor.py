import numpy as np
import pandas as pd


import nn_one

'''We are gonna use the syntax/structure of Andrej Karpathy from his 1st video(bigram model) in the Makemore series.
That is a word, say "emma" is treated as "emma." where dot stands as the ending symbol(NOte that we don't need a starting symbol
for this usecase). Since the longest name has 19 characters, we are going to use a 19*27 neuron input layer
(output is obviously a two neuron layer) so for a 4 letter word, the other 15 characters will be treated as a .'''

def vectorized_result(j):
    #I believe its equivalent to whats called one-hot in pytorch
    """Return a 27-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a character
    (a,b,c...z,.) into a corresponding desired input for the neural
    network."""
    e = np.zeros((27, 1))
    e[j] = 1.0
    return e

def stoi(c):
    if c=='.':
        return 0
    return ord(c)-ord('a') + 1

DATADIR = '/Users/xxxxx' #File location
all_names = pd.read_csv('{}/{}'.format(DATADIR,'Indian-Names.csv'))
xs,ys=[],[]
#This code was just to cross check that a-z are the only characters in all the names
'''super = ""
for names in all_names['Name']:
    #print(super)
    #print(names)
    super+=str(names)
order = sorted(list(set(super)))
print(order)
print(type(all_names['Name'][0]))'''
no = 0
for names in all_names['Name']:
    no+=1
    x = np.zeros((0,1))
    count = 19
    #forced to write str(names) instead of names because the dataset (weirdly) has a name "nan" which python recognizes as a float and not str
    for c in str(names):
        #print(f'{count} characters left')
        count-=1
        e = vectorized_result(stoi(c))
        x = np.concatenate((x,e))
    #In this format dot is appended to the rest of 19 chars, eg: emma is represented as emma......(15 dots)
    while count:
        #print(f'{count} characters left')
        count-=1
        e = vectorized_result(0)
        x = np.concatenate((x,e))
    '''#In this format, emma is represented as emma(15 spaces, full of zeroes)
    if count:
        e = np.zeros((count*27,1))
        x = np.concatenate((x,e))'''
    xs.append(x)
    #print(no)
y_test = []
tr_length = 5971
for gendr in all_names['Gender']:
    if len(ys)==tr_length:
        if gendr=='M':
            y_test.append(0)
        else:
            y_test.append(1)
    else:
        if gendr=='M':
            y = np.array([[1], [0]])
        else:
            y = np.array([[0],[1]])
        ys.append(y)
    
print(f'y-testsize is {len(y_test)}')
print(f'x size is {len(xs)} ysize is {len(ys)}')
print((xs[0]).dtype)
print(ys[0])
x_train = xs[:tr_length]
y_train = ys[:tr_length]
x_test = xs[tr_length:]
training_data = list(zip(x_train,y_train))
test_data = list(zip(x_test,y_test))
print("data generated")
print(len(training_data))
print(len(test_data))
#print(training_data[0])
net = nn_one.Network([513,64,8,2])
net.SGD(30,training_data, 10, 2, test_data=test_data)
