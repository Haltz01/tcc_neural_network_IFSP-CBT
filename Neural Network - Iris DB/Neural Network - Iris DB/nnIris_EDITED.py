print("\n----------------------------------------------------------------------------------\n")
print(" REDE NEURAL ARTIFICIAL - IRIS FLOWER - Versao 4.3")
print("\n----------------------------------------------------------------------------------\n")

print("- IMPORTANDO {\nnumpy,\ntensorflow,\npandas,\nmatplotlib (pyplot, patches),\nkeras.utils (np_utils),\nsklearn.preprocessing (LabelEncoder, OneHotEncoder, StandardScaler),\nsklearn.model_selection (train_test_split, cross_val_score, KFold),\nkeras.wrappers.scikit_learn (KerasClassifier),\nkeras.models (Sequential),\nkeras.layers (Dense, Dropout),\nsklearn.metrics (confusion_matrix),\nitertools\n}")
import numpy as np
#import tensorflow as tf
import pandas as pd
from keras.utils import np_utils
import matplotlib.pyplot as plt #Visualizing the matrixes
import matplotlib.patches as mpatches
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix
import itertools as it

print("- DEFININDO Precisao numerica do Numpy = 5")
np.set_printoptions(precision=5)

print("- IMPORTANDO Base de dados Iris")
iris_dataset = pd.read_csv("iris.data.csv", sep = ",", header = 0, names = ["Sepal length", "Sepal width", "Petal length", "Petal width", "Classification"])
X = iris_dataset.iloc[:, 0:4].values
Y = iris_dataset.iloc[:, 4].values

# ---------------------- PLOTING DATA INFO ----------------------
print("\n- DESENHANDO Graficos a partir das informacoes da base de dados")
#fig1, (data1, data2) = plt.subplots(2)
#fig2, (data3, data4) = plt.subplots(2)

plt.figure(1)
#plt.tight_layout()
sepal_length = mpatches.Patch(color='teal', label='Largura da sepala')
sepal_width = mpatches.Patch(color='blue', label='Comprimento da sepala')
petal_length = mpatches.Patch(color='red', label='Largura da petala')
sepal_width = mpatches.Patch(color='orangered', label='Comprimento da petala')
plt.legend(handles=[sepal_length, sepal_width, petal_length, sepal_width])

plt.title("INFORMAÇAO DAS FLORES")
plt.grid(True)
plt.plot(np.arange(0, X[:, 0].size), X[:, 0], 'o--', c ='teal', linewidth=0.6, markersize=6, mew=0.6, mec="black")
plt.plot(np.arange(0, X[:, 1].size), X[:, 1], 'o--', c = 'blue', linewidth=0.6, markersize=6, mew=0.6, mec="black")
plt.plot(np.arange(0, X[:, 2].size), X[:, 2], 'o--', c = 'red', linewidth=0.6, markersize=6, mew=0.6, mec="black")
plt.plot(np.arange(0, X[:, 3].size), X[:, 3], 'o--', c = 'orangered', linewidth=0.6, markersize=6, mew=0.6, mec="black")
plt.savefig('flowers_info.png')
plt.show()
#data1.plot(np.arange(0, X[:, 0].size), X[:, 1], 'o--', c = 'teal', linewidth=0.6, markersize=5)
#data1.set(title = "Information about the flowers' sepal", ylabel = "Length")
#data2.plot(np.arange(0, X[:, 1].size), X[:, 1], 'o--', c = 'blue', linewidth=0.6, markersize=5)
#data2.set(ylabel = "Width")
#data3.plot(np.arange(0, X[:, 2].size), X[:, 2], 'o--', c = 'red', linewidth=0.6, markersize=5)
#data3.set(title = "Information about the flowers' petal", ylabel = "Length")
#data4.plot(np.arange(0, X[:, 3].size), X[:, 3], 'o--', c = 'orangered', linewidth=0.6, markersize=5)
#data4.set(ylabel = "Width")

# ---------------------- PRE-PROCESSING ----------------------
print("\n- COMECANDO Fase de pre-processamento")

label_encoder_Y = LabelEncoder()
Y[:] = label_encoder_Y.fit_transform(Y[:])

onehot_encoder = OneHotEncoder()
Y = onehot_encoder.fit_transform(Y.reshape(-1, 1)).toarray()

#Y = np_utils.to_categorical(Y) # Multiclass classification problem

print("- SEPARANDO Elementos da base de dados (train_x, train_y, test_x, test_y)")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, shuffle=True, random_state = 7)
print("- FINALIZANDO Fase de pre-processamento")

# ---------------------- STANDARDIZING ----------------------
print("\n- COMECANDO Fase de padronizacao")

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print("- ENDING Standardizing phase")

# ---------------------- EVALUATION ----------------------
print("\n- COMECANDO Fase de avaliacao")

batch_sizes = np.array([5, 10])
epochs = np.array([100, 200])
all_means = []
all_variances = []
all_accuracies = []
verbose_op = 1

def build_classifier():
    classifier = Sequential() 
    classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = 4)) #First hidden layer + INPUTS!
    #classifier.add(Dropout(0.1))
    #classifier.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'relu')) #Second hidden layer
    classifier.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'softmax')) #Output layer
    classifier.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"]) #categorical_crossentropy, binary_crossentropy or mean_squared_error
    return classifier

kfold = KFold(n_splits = 10, shuffle = True, random_state = 7)

print("- BUILDING Neural network classifier with KerasClassifier (default options: optimizer = adam, loss= categorical_crossentropy, batch_size = 5, epochs = 100)")
print("- TRAINING Classifiers [cv = KFold(n_splits = 10, shuffle = True, random_state = 7)]")

for a in range(0, epochs.size):
    for b in range(0, batch_sizes.size):
        print("- AVALIANDO Classifier de numero: [" + str(a) + "][" + str(b) + "]")
        print("- DEFININDO batch_size = " + str(batch_sizes[b]) + " - epochs = " + str(epochs[a])) 
        classifier = KerasClassifier(build_fn = build_classifier, batch_size = batch_sizes[b], epochs = epochs[a], verbose = verbose_op) 
        #DROPOUT = 1/2 if OVERFITTING!!
        print("- CALCULANDO precisoes, medias e variancias usando [cross_val_score]")
        accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = kfold) #10x10 groups of evaluation (1 in each for testing, 9 for training)
        all_accuracies.append(accuracies)
        all_means.append(accuracies.mean()) #"média"
        all_variances.append(accuracies.std())  #standard deviation
        #input("PRESS ANY BUTTON TO CONTINUE")

print("\nPRECISOES: ")
i = 1
for row in all_accuracies:
    print(str(i) + "º) " + str(row))
    i+=1

print("\nMEDIAS E VARIANCIAS: ")
#i = 1
for i in range(0, np.array(all_means).size):
    print((str(i+1) + "º) Mean: {0:.5f} --- Variance: {1:.5f}").format(all_means[i], all_variances[i]))
    if i == 0:
        best_mean = [all_means[i], i]
        best_var = [all_variances[i], i] #related to mean, not the 'pure' best one
    elif best_mean[0] < all_means[i] and all_variances[i] < (best_var[0]+0.18):
        best_mean = [all_means[i], i]
        best_var = [all_variances[i], i]
    i+=1

'''
print("\nVARIANCE: ")
j = 1
for row in all_variances:
    print((str(j) + "º) {0:.5f}").format(row))
    if j == 1:
        best_var = [row, j]
    elif best_var[0] > row:
        best_var = [row, j]
    j+=1
'''

print("\nMELHORES RESULTADOS:")
print("MELHOR MEDIA (BASEADA NA VARIANCIA): " + str(best_mean[0]) + " - Contador: " + str(best_mean[1]))
#print("BEST VARIANCE: " + str(best_var[0]) + " - Counter: " + str(best_var[1]))
print("- FINALIZANDO Fase de avaliacao")

a = -1
b = -1
for x in range (0, (best_mean[1]+1)):
    if x < epochs.size:
        a += 1
    else:
        b += 1
        
epoch = epochs[a]
batch_size = batch_sizes[b]
valid = False

while not valid:
	resp = input(("- PERGUNTA! Voce quer construir a rede neural \ncom a melhor configuracao encontrada? (batch size = {0} e epochs = {1})?").format(batch_size, epoch))
	if resp.lower().startswith("s"):
		print("- CONTINUANDO Programa")
		valid = True;
	elif resp.lower().startswith("n"):
		valid = True;
		print("- FINALIZANDO Programa")
		exit()

classifier = build_classifier()
print(("- COMPILANDO Rede neural artificial (epochs = {0} - batch_size = {1})").format(epoch, batch_size))
print("- DEFININDO Compilador: optimizer = adam, loss = categorical_crossentropy, metrics = [accuracy]")
classifier.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
print("- TREINANDO Rede neural artificial (fitting)")
classifier.fit(X_train, Y_train, batch_size = batch_size, epochs = epoch)

# ---------------------- PREDICTION ----------------------
print("\n- COMECANDO Fase de testes (previsoes)")

def prepare4pred(X):
    Y = np.array([])
    for line in X:
        if line[0] > line[1] and line[0] > line[2]:
            Y = np.append(Y, 0)
        elif line[1] > line[0] and line[1] > line[2]:
            Y = np.append(Y, 1)
        elif line[2] > line[0] and line[2] > line[1]:
            Y = np.append(Y, 2)
        else:
            np.append(Y_pred_final, 3)
        
    return Y

print("- CALCULANDO saidas dos elementos em [X_test]")
Y_pred = classifier.predict(X_test)

Y_pred = prepare4pred(Y_pred)
Y_test = prepare4pred(Y_test)

print("- CONSTRUINDO Matriz comparativa de saidas (calculado X esperado) para obter a precisao do teste")
#Making the Confusion Matrix: calculate the the real accuracy obtained from the test
conf_matrix = confusion_matrix(Y_test, Y_pred)
percent_cm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

print("- FINALIZANDO Fase de testes (previsoes)")

# ---------------------- PLOTING ----------------------
print("\n- COMECANDO Fase de construcao grafica da matriz")

def plot_confusion_matrix(cm, title, classes=["Setosa", "Versicolor", "Virginica"]):
    
    np.set_printoptions(precision=2)
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Purples) 
    #https://matplotlib.org/gallery/images_contours_and_fields/interpolation_methods.html
    plt.title(title)
    plt.colorbar() #adds colorbar to plot
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0) #subtitles - description
    plt.yticks(tick_marks, classes)

    th_color = cm.max() / 2.
    for i, j in it.product(range(cm.shape[0]), range(cm.shape[1])): #cartesian product # for inside for (nested loop)
        plt.text(j, i, format(cm[i, j], ".2f"), horizontalalignment="center",  color="white" if cm[i, j] > th_color else "black")

    plt.tight_layout()
    plt.ylabel('Saidas reais')
    plt.xlabel('Saidas esperadas')

print("- CONSTRUINDO Matriz graficamente (nao-normatizada)")
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(conf_matrix, "Confusion matrix - Quantitativa")
plt.savefig('confusion_matrix_quantifier.png')

print("- CONSTRUINDO Matriz graficamente (normatizada)")
# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(percent_cm, "Confusion matrix - Percentual")
plt.savefig('confusion_matrix_percentage.png')

plt.show()
print("- FINALIZANDO Fase de construcao grafica da matriz")


