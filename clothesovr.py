import numpy as np
import pickle
from keras.datasets import fashion_mnist
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix
from keras.preprocessing import image
from PIL import Image

#OvR strategy
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
fashions_train = []
fashions_test = []
for fashion_train in x_train:
    fashions_train.append(fashion_train.flatten())
for fashion_test in x_test:
    fashions_test.append(fashion_test.flatten())
fashions_train = np.array(fashions_train)
fashions_test = np.array(fashions_test)
#print(fashion_test)
fashion_mnist_classifier = OneVsRestClassifier(LogisticRegression(verbose=1, max_iter = 10))
fashion_mnist_classifier.fit(fashions_train, y_train)
conf_matrix = confusion_matrix(y_test, fashion_mnist_classifier.predict(fashions_test))
print("Confusion_matrix:")
print(conf_matrix)
print('Score %s' % fashion_mnist_classifier.score(fashions_test, y_test))


#Multiple classes with one classifier
fashion_mnist_classifier = LogisticRegression(verbose=1, max_iter=3, multi_class="multinomial", solver="sag")
fashion_mnist_classifier.fit(fashions_train, y_train)
conf_matrix = confusion_matrix(y_test, fashion_mnist_classifier.predict(fashions_test))
print("Confusion_matrix:")
print(conf_matrix)
fashion_mnist_classifier.score(fashions_test, y_test)
print('Score %s' % fashion_mnist_classifier.score(fashions_test, y_test))
pickle.dump(fashion_mnist_classifier, open('fashion_mnist_classifier.model', 'wb'))
fashion_mnist_classifier__from_file = pickle.load(open('fashion_mnist_classifier.model', 'rb'))
conf_matrix = confusion_matrix(y_test, fashion_mnist_classifier__from_file.predict(fashions_test))
print("Confusion_matrix:")
print(conf_matrix)
print(fashions_test.__len__)
print('Score %s' % fashion_mnist_classifier__from_file.score(fashions_test, y_test))
#Checking on own image
img = Image.open('koszulka.jpg').convert('LA')
img.save('greyscale1.png')
image_file = 'greyscale1.png'
img = image.load_img(image_file, target_size=(28, 28), grayscale=True, color_mode="grayscale")
x = image.img_to_array(img)
y = x.flatten().reshape(1,-1)
print("Zakwalifikowano jako:")
print(fashion_mnist_classifier__from_file.predict(y))   #Uzyto obrazka koszulki, zakwalifikowane prawidłowo