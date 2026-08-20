## BASIC NEURAL NETWORK
import numpy as np

## INPUT
X = np.array([1,2])

## INITIAL WEIGHTS
W = np.array([0.5,0.3])

## BIAS
b = 0.1

## TARGET OUTPUT
y = 1

## ACTIVATION FUNCTION
def sigmoid(X):
    return 1/(1+np.exp(-X))

## NEURAL NETWORKS   ==> FORWARD PROPOGATION
print("\nFORWARD PROPOGATION")
z = np.dot(X,W)+b
output = sigmoid(z)
print("Output = ",output)

error = y-output
print("error =",error)
if output >= 0.5:
    output=1
else:
    output=0
print("Output = ",output)

## Forward Propogation
## Input ==> Weighted *input (Z) ==> Activation Functio ==> Output ==> Loss

## BACKWARD PROPOGATION
## Loss ==> Output ==> Activation ==> Z ==> Weights and Bias

print("\nBACKWARD PROPOGATION")
sigmoid_derivative = output*(1-output)
delta = error*sigmoid_derivative

learning_rate = 0.1

W = W + learning_rate*delta*X
b = b + learning_rate*delta
# m = m + learning_rate*dm
print("Updated Weights = ",W)
print("Update Bias =",b)

## Forward Propogation
print("\nForward Propogation after Training")
z = np.dot(X,W)+b
new_output = sigmoid(z)
print("Output after Training =",new_output)