Differences from previous model:

As the training images are less (currency note will not change in any manner unlike other objects), it is safe to assume that only for a higher accuracy , the image is a 2000 ruppee.

So, in program :


Threshold > 0.8 ---- 2000 ruppee note
Added alpha - 0.001, 
Added bias in hidden layers (+5) ( Not adding this also results good enough )
Only 3 neurons

PS - Adding neurons causes the accuracy( in test images ) to become less 
