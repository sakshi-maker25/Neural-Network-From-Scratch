import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

hls = 5  # size of hidden layer
N = 500
ftrgridpoints = 24  # for the decision map

# X, y = make_moons(n_samples=N, noise=0.1, random_state=42) [random_state=42 = If you run it again using the same random state, you get the same dataset.]
# np.save('moonDataX', X)
# np.save('moonDatay', y)

X = np.load('moonDataX.npy')
y = np.load('moonDatay.npy')

xmaxv, xminv = 2, -2  # scale feature data to [-2, 2]x[-2, 2]
X_min = X.min(axis=0) #axis = 0 means Go down the rows and calculate separately for each column
X_max = X.max(axis=0)
Xscld = (X - X_min) / (X_max - X_min) * (xmaxv - xminv) + xminv
X = Xscld #From this point onwards whenever the program uses X it uses the scaled version.

y[y == 0] = -1  # relabel 0 to -1 since we are using tanh

# Plot the data
plt.figure(figsize=(6, 6))
plt.scatter(X[y == -1, 0], X[y == -1, 1], c='red', label='Class -1', alpha=0.7, edgecolors='k')
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', label='Class 1', alpha=0.7, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('The generated "two half-moon" dataset')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('dataX.png')

def linear_layer(A):
    def ffwd(x):  # forward calculation
        xext = np.append(x, 1)  # append 1 because bias is included in A
        return np.tanh(A @ xext)

    def ffwd_dsh(x):  # derivative of tanh
        xext = np.append(x, 1)
        z = A @ xext
        return 1 - np.tanh(z)**2

    return ffwd, ffwd_dsh

np.random.seed(42)

Aone = np.random.randn(hls, 2+1)  # matrix shape 5x3
Atwo = np.random.randn(1, hls+1)  # matrix shape 1x6

(LLone, LLone_dsh) = linear_layer(Aone)
(LLtwo, LLtwo_dsh) = linear_layer(Atwo)

# Training settings
learning_rate = 0.01
epochs = 30

# Stochastic Gradient Descent
for epoch in range(epochs):

    # randomly change the order of the data points
    indices = np.random.permutation(N)

    # use one data point at a time
    for i in indices:
        x = X[i]
        y_true = y[i]

        # FORWARD PASS
        hidden = LLone(x)  # outputs of the 5 hidden neurons
        yhat = LLtwo(hidden)  # prediction

        # BACKPROPAGATION

        # error signal for second layer
        delta2 = 2 * (yhat - y_true) * LLtwo_dsh(hidden)

        # gradient for second layer
        hidden_ext = np.append(hidden, 1)
        grad_Atwo = np.outer(delta2, hidden_ext)

        # error signal for first layer
        delta1 = (Atwo[:, :-1].T @ delta2) * LLone_dsh(x)

        # gradient for first layer
        x_ext = np.append(x, 1)
        grad_Aone = np.outer(delta1, x_ext)

        # update the weights
        Aone -= learning_rate * grad_Aone
        Atwo -= learning_rate * grad_Atwo

    # Calculate MSE and accuracy after one epoch
    total_loss = 0
    correct = 0

    for j in range(N):
        prediction = LLtwo(LLone(X[j])).item()
        total_loss += (prediction - y[j])**2

        if prediction >= 0:
            predicted_label = 1
        else:
            predicted_label = -1

        if predicted_label == y[j]:
            correct += 1

    mse = total_loss / N
    accuracy = correct / N

    # Print epochs
    if epoch == 0 or (epoch + 1) % 10 == 0:
        print("Epoch:", epoch + 1)
        print("MSE:", mse)
        print("Accuracy:", accuracy * 100, "%")
        print()

print("Training finished")
print("Final MSE:", mse)
print("Final Accuracy:", accuracy * 100, "%")

# Decision map
ftr1range = np.linspace(xminv, xmaxv, ftrgridpoints)
ftr2range = np.linspace(xminv, xmaxv, ftrgridpoints)

decisionregion = np.zeros((ftrgridpoints, ftrgridpoints))

for ftr1id in range(ftrgridpoints):
    for ftr2id in range(ftrgridpoints):
        f1f2 = np.array([ftr1range[ftr1id], ftr2range[ftr2id]])
        yhat = LLtwo(LLone(f1f2))
        decisionregion[ftr2id, ftr1id] = yhat.item()

plt.figure(figsize=(6, 6))
plt.title('Decision Map')
plt.imshow(decisionregion, cmap='cool', extent=[xminv, xmaxv, xminv, xmaxv], origin='lower')
plt.savefig('ini-decision-map.png')
plt.show()

# print("Input point:", x)
# print("True label:", y_true)
# print("Hidden output:", hidden)
# print("Network prediction:", yhat)
# print("Loss:", loss)
# print("x shape:", x.shape)
# print("hidden shape:", hidden.shape)
# print("yhat shape:", yhat.shape)
# print("Aone shape:", Aone.shape)
# print("Atwo shape:", Atwo.shape)
# print("grad_Aone shape:", grad_Aone.shape)
# print("grad_Atwo shape:", grad_Atwo.shape)