import numpy as np
from sklearn.linear_model import LinearRegression

#Ejercicio 1: Se tiene un conjunto de datos que representa la cantidad de horas de
#estudio de un grupo de estudiantes y sus correspondientes calificaciones en un examen.
#Utiliza la regresión lineal para crear un modelo que pueda predecir la calificación de un
#estudiante en función de las horas de estudio. Luego, utiliza el modelo para predecir la 
#calificación de un estudiante que haya estudiado 5 horas.

# artificial data
num_samples=99
num_hours=np.random.randint(low=3,high=12,size=num_samples)
grades = 35 + 4 * num_hours + np.random.normal(0, 5, size=num_samples)

# Stack
X = np.column_stack((num_hours,))

print (X)
# Model
model = LinearRegression()
model.fit(X, grades)

#Prediccion

new_hour = np.array([[15]])

prediction=model.predict(new_hour)

print(prediction)




# Ejercicio 2: Estás trabajando en una empresa de comercio electrónico y 
# tienes datos de ventas mensuales en relación con los gastos en publicidad en diferentes 
# canales (por ejemplo, redes sociales, televisión, radio, etc.). Utiliza la regresión lineal 
# para construir un modelo que relacione los gastos en publicidad con las ventas mensuales.
#  Después, utiliza el modelo para predecir las ventas del próximo mes en función de los gastos
#  en publicidad que se planean realizar

# artificial data
num_samples2=49
tv_advertising=np.random.randint(low=20000,high=100000,size=num_samples2)
social_media_advertising=np.random.randint(low=20000,high=50000,size=num_samples2)
radio_advertising=np.random.randint(low=1000,high=10000,size=num_samples2)
newspaper_advertising=np.random.randint(low=500,high=10000,size=num_samples2)
sales = 90000 + 2 * tv_advertising + 2 * social_media_advertising + 0.5 * radio_advertising + 0.6 * newspaper_advertising + np.random.normal(0, 50000)



# Stack
X2 = np.column_stack((tv_advertising,radio_advertising,social_media_advertising,newspaper_advertising))

print (X2)
# Model
model = LinearRegression()
model.fit(X2, sales)

#Prediccion
next_tv_advertising = 100000
next_social_media_advertising = 35000
next_radio_advertising = 5000
next_newspaper_advertising = 2000

next_sales = np.array([[next_tv_advertising,next_social_media_advertising,next_radio_advertising,next_newspaper_advertising]])
prediction2=model.predict(next_sales)

print(prediction2)
