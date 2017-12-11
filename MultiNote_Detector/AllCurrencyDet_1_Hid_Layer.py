import numpy as np  

import cv2
import numpy as np


img=cv2.imread ("web_10_thumb.jpg")
img2=cv2.imread("web_20_thumb.jpg")

img3=cv2.imread("web_50_thumb.jpg")
img4=cv2.imread("web_100_thumb.jpg")
img5=cv2.imread("web_500_thumb.jpg")
img6=cv2.imread("web_1000_thumb.jpg")
img7=cv2.imread("web_2000_thumb.jpg")

w1, h1 = img.shape[:2]
#print width,height
w2, h2 = img2.shape[:2]
w3, h3 = img3.shape[:2]
w4, h4 = img4.shape[:2]
w5, h5 = img5.shape[:2]
w6, h6 = img6.shape[:2]
w7, h7 = img7.shape[:2]

print w1,h1
print w2,h2
print w3,h3
print w4,h4
print w5,h5
print w6,h6
print w7,h7

a1=np.array(img)
a2=np.array(img2)
a3=np.array(img3)
a4=np.array(img4)
a5=np.array(img5)
a6=np.array(img6)
a7=np.array(img7)

#cv2.imshow('img',img)
#cv2.imshow('img2',img2)


b1=np.zeros(w2*h1*3+1)
b2=np.zeros(w2*h2*3+1)
b3=np.zeros(w2*h2*3+1)
b4=np.zeros(w2*h2*3+1)
b5=np.zeros(w2*h2*3+1)
b6=np.zeros(w2*h2*3+1)
b7=np.zeros(w2*h2*3+1)

n=0

for i in xrange(0,3):
    for j in xrange(0,w7):
        for k in xrange(0,h3):
            
    

            b1[n] = a1[j][k][i]*0.00392156862
            b2[n] = a2[j][k][i]*0.00392156862
            b3[n] = a3[j][k][i]*0.00392156862
            b4[n] = a4[j][k][i]*0.00392156862
            b5[n] = a5[j][k][i]*0.00392156862
            b6[n] = a6[j][k][i]*0.00392156862
            b7[n] = a7[j][k][i]*0.00392156862
            #print "Sig" ,b[n]
            n+=1

b1[n]=1
b2[n]=1
b3[n]=1
b4[n]=1
b5[n]=1
b6[n]=1
b7[n]=1

#print b


def nonlin(x, deriv=False): 
    if(deriv==True):
        return (x*(1-x))
    
    return 1/(1+np.exp(-x))  

#input data
#print b1,b2

X = np.array([b1,b2,b3,b4,b5,b6,b7])


y = np.array([[0.5],[0.58],[0.66],[0.75],[0.83],[0.91],[0.99]])


np.random.seed(1)
print X

n+=1
print n
#synapses
syn0 = np.random.random((n,10)) *0.00001  # 3x4 matrix of weights ((2 inputs + 1 bias) x 4 nodes in the hidden layer)
syn1 = np.random.random((10,1))    # 4x1 matrix of weights. (4 nodes x 1 output) - no bias term in the hidden layer.
i=0
print syn0
for j in range(1000):  
    
    # Calculate forward through the network.
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))
    #print l0
    #print l1
    #print l2
    # Back propagation of errors using the chain rule. 
    l2_error = y - l2
    if(j % 10) == 0:   # Only print the error every 1000 steps, to save time and limit the amount of output. 
        print("Error: " + str(np.mean(np.abs(l2_error))))
        print l2
        
    l2_delta = l2_error*nonlin(l2, deriv=True)
    
    l1_error = l2_delta.dot(syn1.T)
    
    l1_delta = l1_error * nonlin(l1,deriv=True)
    
    #update weights (no learning rate term)
    syn1 += l1.T.dot(l2_delta)*0.001
    syn0 += l0.T.dot(l1_delta)*0.001
    
    
    
print("Output after training")
print l2
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
#
#    print l2[i]

'''print "Testing 1"
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
'''

print "Testing"
n=0
imtest =cv2.imread("Testt3.jpg")
atest=np.array(imtest)
btest=np.zeros (w1*h2*3+1)

for i in xrange(0,3):
    for j in xrange(0,w7):
        for k in xrange(0,h3):
            
    

            btest[n] = atest[j][k][i]*0.00392156862
            #print "Sig" ,b[n]
            n+=1

btest[n]=1

X=np.array([btest,btest,btest,btest,btest,btest,btest])

l0 = X
l1 = nonlin(np.dot(l0, syn0))
l2 = nonlin(np.dot(l1, syn1))

print l2
p=l2[0]

if p<=0.535 :
    print "10 Bucks"
elif p<=0.6 and p>=0.58:
    print "20 Bucks"
elif p<=0.67 and p>=0.65:
    print "50 Bucks"
elif p<=0.77 and p>=0.75:
    print "100 Bucks"
elif p<=0.84 and p>=0.82:
    print "500 Bucks"
elif p<=0.92 and p>=0.90:
    print "1000 Bucks"
elif p>=0.98:
    print "2000 Bucks"
else:
    print "Not a Currency"


