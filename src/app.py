#from utils import db_connect
#engine = db_connect()

import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Predecir el coste del seguro médico de una persona

# Cargo datos
total_data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/linear-regression-project-tutorial/main/medical_insurance_cost.csv")

# Elimino duplicados
if total_data.duplicated().sum():
    total_data = total_data.drop_duplicates()
print(total_data.shape)
total_data.head()

# Borro datos nulos
total_data.isnull().sum().sort_values(ascending=False)

'''
Haciendo el análisis univariante, multivariante y las matrices de correlación, he llegado a la conclusión de que escojo las siguientes variables como predictoras:
"age", "smoker_n", "bmi"
'''
# División del conjunto en train y test
total_data["smoker_n"] = pd.factorize(total_data["smoker"])[0] #Factorización por ser categórica
num_variables = ["age", "smoker_n", "bmi"]

X = total_data.drop("charges", axis = 1)[num_variables]
y = total_data["charges"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Escalado de variables predictoras
scaler = MinMaxScaler()
scaler.fit(X_train)

X_train_scal = scaler.transform(X_train)
X_train_scal = pd.DataFrame(X_train_scal, index = X_train.index, columns = num_variables)

X_test_scal = scaler.transform(X_test)
X_test_scal = pd.DataFrame(X_test_scal, index = X_test.index, columns = num_variables)

# Entrenamiento del modelo
model = LinearRegression()
model.fit(X_train_scal, y_train)

# Cálculo de intercepto y coeficientes (en este caso 3 porque tengo 3 variables predictoras)
print(f"Intercepto (a): {model.intercept_}")
print(f"Coeficientes (b): {model.coef_}")

y_pred = model.predict(X_test_scal)

# Cálculo de métricas para evaluar la eficacia del entrenamiento
print(f"Error cuadrático medio: {mean_squared_error(y_test, y_pred)}")
print(f"Coeficiente de determinación: {r2_score(y_test, y_pred)}")
