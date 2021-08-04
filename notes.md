# Book
Make Your Own Neural Network

# Very Basic Predictor
KMs * c = Miles.
Initialise random c

Calculate and get an output:
E = desired output - actual output
Adjust c to minimise the error.

# Very Basic Classifier
ax = y
Initalise random a (a = 15)
Try training size (x: 2, y:3)
15 * 2 = 30

How to calculate adjustment in A:
Actual:
    y = ax
Target:
    t = (a + deltaa)x

Target t - actual y = error E
    (a + deltaa)x - ax = E
    a*x + deltaa*x - ax = E
    (deltaa)x = E
    deltaa = E/x

Try training set again (adjust a little because we dont want the line to through point. Just below or above)

Try training size (x: 2, y:2.9)
15 * 2 = 30

E = t - a = 2.9 - 30 = -27,1
deltaa = -27.1/2 = -13.55

Adjust
(15 - 13.55)x = y
1.05x = y

Try again
1.45 * 2 = y = 2.9


# Learning rate
When new a is calculated it changes everything and negate earlier training example.
Learning rate L to preserve some of the adjustmens from earlier traning examples

L * (aX) = Y

# Neural Network
Calculating neural networks is like matrix multiplication
(weights of input 1, weights of input 2)
(w11, w21) * (input 1)
 w12, w22     input 2

X = W*I
Matrices are shown in bold and big letters

Remeber activiation function:
Output O = sigmoid(X)

Matrices are a data stucture that computer can do operations effectiently on

# Back probagation
Find error and split errors over weight.
Change the weights with most influence because it was them who made the error the most

Error of 4 on w1 = 1, w2 = 3

# Back probagation in multiple layers
There is no target, so to ge the error, the error from the links are combined together to make an error for hidden nodes

# Gradient decent
Smaller steps propertial to the gradient, not to overshoot it