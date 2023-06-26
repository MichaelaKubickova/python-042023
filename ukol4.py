import pandas
import numpy
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import export_graphviz
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from six import StringIO
from IPython.display import Image  
import pydotplus
from pydotplus import graph_from_dot_data
import os
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"

data = pandas.read_csv("ukol_04_data.csv")
print(data.head())

##BOD1
##Rozdělení dat dle vstupní proměnné ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "duration", "campaign", "pdays", "previous", "poutcome"] a výstupní proměnná ["y" = termínovaný účet ANO/NE]
categorical_columns = ["job", "marital", "education", "default", "housing", "loan", "contact", "campaign", "poutcome"]
numeric_columns = ["age", "balance", "duration", "pdays", "previous"]
numeric_data = data[numeric_columns].to_numpy()

y = data["y"]
##Pomocí OneHotEncoder převedeme na pole
encoder = OneHotEncoder()
encoded_columns = encoder.fit_transform(data[categorical_columns])
encoded_columns = encoded_columns.toarray()

X = numpy.concatenate([encoded_columns, numeric_data], axis=1)
#print(X)

##Získáme názvy sloupců
encoder.get_feature_names_out()
print(X)
##Rozdělíme data na trénovací a testovací
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
##Vytvoříme klasifikátor využívající algoritmus rozhodovacího stromu - strom může mít maximálně 4 patra (kořen nepočítáme)
clf = DecisionTreeClassifier(max_depth=4)
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
##Uložíme strom jako obrázek pomocí metody write_png() a přidáme popisky 
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data, filled=True, feature_names=list(encoder.get_feature_names_out()) + numeric_columns, class_names=["no", "yes"])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('tree.png')
Image(graph.create_png())
##Vytvoření matice záměn
cm_display = ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
cm_display.plot()
plt.show() 
##Výpočet accuracy
print(accuracy_score(y_test, y_pred))

##BOD2
#Precision: Tato metrika penalizuje označení zákazníka, který nemá zájem o termínovaný účet za zákazníka, který má zájem. Čím více zákazníků, kteří nemají zájem o 
##termínovaný účet označíme za zájemce, tím má metrika menší hodnotu. Metrika nepočítá s tím, kolik zájemců jsme označili za nezájemce.

##BOD3
print(precision_score(y_test, y_pred, pos_label="yes"))


##BOD4
scaler = StandardScaler()
numeric_data = scaler.fit_transform(data[numeric_columns])

encoder = OneHotEncoder()
encoded_columns = encoder.fit_transform(data[categorical_columns])
encoded_columns = encoded_columns.toarray()

X = numpy.concatenate([encoded_columns, numeric_data], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

##KNeighborsClassifier
model_1 = KNeighborsClassifier()
params_1 = {"n_neighbors": range(3, 15, 23)}

clf_1 = GridSearchCV(model_1, params_1, scoring="accuracy")
clf_1.fit(X, y)

print(clf_1.best_params_)
print(round(clf_1.best_score_, 2))

##BOD5
##Linear SVC
model_2 = SVC(kernel="linear")
params_2 = {"decision_function_shape": ["ovo", "ovr"]}

clf_2 = GridSearchCV(model_2, params_2, scoring="accuracy")
clf_2.fit(X, y)

print(clf_2.best_params_)
print(round(clf_2.best_score_, 2))

##BOD6
#nejlépe výchází metrika pro Linear SVC 66%


