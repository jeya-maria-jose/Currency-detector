import numpy as np  

import cv2
import numpy as np


img=cv2.imread ("Test3.jpg")
img2=cv2.imread("2000.jpg")

w1, h1 = img.shape[:2]
#print width,height
w2, h2 = img2.shape[:2]
print w2,h2

a1=np.array(img)
a2=np.array(img2)
cv2.imshow('img',img)
cv2.imshow('img2',img2)


b1=np.zeros(w2*h1*3+1)
b2=np.zeros(w2*h2*3+1)
b3=np.zeros(w2*h2*3+1)
n=0
for i in xrange(0,3):
    for j in xrange(0,w2):
        for k in xrange(0,h2):
            
    

            b1[n] = a1[j][k][i]*0.00392156862
            b2[n] = a2[j][k][i]*0.00392156862
            b3[n] = 0
            
            #print "Sig" ,b[n]
            n+=1

b1[n]=1
b2[n]=1
b3[n]=1
#print b


def nonlin(x, deriv=False): 
    if(deriv==True):
        return (x*(1-x))
    
    return 1/(1+np.exp(-x))  

#input data
#print b1,b2

X = np.array([b2])


y = np.array([[1]])



np.random.seed(1)
print X

n+=1
print n
#synapses
syn0 = np.random.random((n,3))   # 3x4 matrix of weights ((2 inputs + 1 bias) x 4 nodes in the hidden layer)
syn1 = np.random.random((3,1))  # 4x1 matrix of weights. (4 nodes x 1 output) - no bias term in the hidden layer.


for j in range(1000):  
    
    # Calculate forward through the network.
    l0 = X
    l1 = nonlin(np.dot(l0, syn0)+5)
    l2 = nonlin(np.dot(l1, syn1)+5)
    
    # Back propagation of errors using the chain rule. 
    l2_error = y - l2
    if(j % 1000) == 0:   # Only print the error every 1000 steps, to save time and limit the amount of output. 
        print("Error: " + str(np.mean(np.abs(l2_error))))
        
    l2_delta = l2_error*nonlin(l2, deriv=True)
    
    l1_error = l2_delta.dot(syn1.T)
    
    l1_delta = l1_error * nonlin(l1,deriv=True)
    
    #update weights (no learning rate term)
    syn1 += l1.T.dot(l2_delta)*0.001
    syn0 += l0.T.dot(l1_delta)*0.001
    
print("Output after training")

#for i in range(2):
 #   print(l2[i])
#print syn1,syn0
#print b1,b2,b3
'''if all(b1)==all(b2):
    print "wtf"
    print b1
    print b2
else:
    print "good"
'''
print l2
print "Testing 1"
X = np.array([b1])

l0 = X
l1 = nonlin(np.dot(l0, syn0))
l2 = nonlin(np.dot(l1, syn1))

print l2

print "Testing 2"
X = np.array([b3])

l0 = X
l1 = nonlin(np.dot(l0, syn0))
l2 = nonlin(np.dot(l1, syn1))

print l2

cv2.waitKey(0)