import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neural_network import MLPClassifier      
## Multi-Layer Perceptron
from sklearn.neighbors import KNeighborsClassifier

## DATA PREPROCESSING  ==> STARTS 

# digits = datasets.load_digits()
# print(digits.DESCR)
# print(digits.images)         ## features
# print(digits.target)         ## LABELS
# print(len(digits.target))    ## show instances  ## 1797
# print(digits.target.shape)       ## (1797,)
# print(digits.images.shape)       ## (1797, 8, 8)

# images_with_labels = list(zip(digits.images,digits.target))
# print(images_with_labels[:2])

# for index, (image,label) in enumerate(images_with_labels[:6]):
#     plt.subplot(2, 3, index + 1)              ## (row,column,index)
#     plt.imshow(image,cmap=plt.cm.gray_r)
#     plt.title("Training : %i" %label)
# plt.show()

# n_samples = len(digits.images)
# data = digits.images.reshape(n_samples,-1)
# print(digits.images.shape)        ## (1797, 8, 8)
# print(data.shape)                 ## (1797, 64)
# data = data/255.
# print(data)

X,y =datasets.load_digits(return_X_y=True)
# print(X.shape)                      ## (1797, 64)
# print(y.shape)                      ## (1797,)
# print(X[:2])
X=X/255.
# print(X[:2])
## DATA PREPROCESSING  ==> ENDS

model= MLPClassifier(hidden_layer_sizes=(50,),max_iter=300,random_state=1,verbose=1,learning_rate_init=.1)
model.fit(X,y)
print("MLP Classifier : ",model.score(X,y))

# KNN Classifier
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X, y)
print("KNN Classifier :", knn_model.score(X, y))
