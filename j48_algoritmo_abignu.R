# en este script vamos a ejecutar el algoritmo j48 sobre el dataset de acuiferos en Tanzania, realizado por Agustin Bignu
require(tidyverse)
require(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(Hmisc)
library(RWeka) # para c4.5 J48

# cargamos datasets
camino = 'C:/Users/agustin/Google Drive/Master AI/Asignaturas/Mineria de Datos Preprocesamiento y Clasificacion/TrabajoFinal/'

datos = read.csv(paste(camino,'clean.csv', sep = ''), header = T, sep = ',')
test = read.csv(paste(camino,'clean_test.csv', sep = ''), header = T, sep = ',')

submissionformat = read.csv(paste(camino,'SubmissionFormat.csv', sep = ''), header = T, sep = ',')

#view(datos)
# vamos a convertir los meses para tener otra variable mas
# date_recorded_offset_days_train <- as.numeric(as.Date("2014-01-01") - as.Date(datos$date_recorded))
# date_recorded_month_train <- factor(format(as.Date(datos$date_recorded), "%b"))
# datos <- datos[, -which(names(datos) == "date_recorded")]
# datos <- cbind(datos, date_recorded_offset_days)
# datos <- cbind(datos, date_recorded_month)

# date_recorded_offset_days_test <- as.numeric(as.Date("2014-01-01") - as.Date(test$date_recorded))
# date_recorded_month_test <- factor(format(as.Date(test$date_recorded), "%b"))
# test <- test[, -which(names(test) == "date_recorded")]
# test <- cbind(test, date_recorded_offset_days)
# test <- cbind(test, date_recorded_month)

# write.csv(date_recorded_month_train, paste(camino,'mes_train.csv', sep = ''))
# write.csv(date_recorded_month_test, paste(camino,'mes_test.csv', sep = ''))

# leo meses para usarlos como variable a ver si mejora el modelo (no mejora)
mes_train = read.csv(paste(camino,'mes_train.csv', sep = ''))
mes_test = read.csv(paste(camino,'mes_test.csv', sep = ''))

# los meto en el dataset
# datos$mes = mes_train$x
# test$mes = mes_test$x

# elimino algunas variables que no quiero tener por motivos de que el modelo no mejora con ellas
datos$X = NULL # eliminamos esta columna
id_train = datos$id
datos$id = NULL
datos$lga = NULL
datos$working_years = NULL
datos$date_recorded = NULL # para random forest
datos$subvillage = NULL # lo saco por razones de que disminuye la prediccion

test$X = NULL
test$lga = NULL
id_test = test$id
test$id = NULL
test$working_years = NULL
test$date_recorded = NULL # para random forest
test$subvillage = NULL

names(datos)
names(test)

dim(test) # 59400 filas y 24 columnas
dim(datos)

# hay que limpiar algunas variables con muhas categorias --> wranglers
str(datos) # district code y lga aportan la misma informacion, por lo que lga la saco

# probamos primer modelo de J48
# primero separamos los datos
## 75% of the sample size
smp_size <- floor(0.75 * nrow(datos))

#set.seed(123)
train_ind <- sample(seq_len(nrow(datos)), size = smp_size)

train = datos[train_ind,]
test_2 = datos[-train_ind,]


fit = J48(status_group~., data = train, control = Weka_control(M = 2, C = 0.31))
preds = predict(fit, test_2)

class.pred <- table(preds, test_2$status_group)
1-sum(diag(class.pred))/sum(class.pred) # saco el classification error rate; 0.2137

summary(fit)
# accuracy
sum(preds == test_2$status_group) / length(preds)

# cost sensitive (sale igual al de J48)
fit2 = CostSensitiveClassifier(status_group ~ ., data = train, control = Weka_control('cost-matrix' = matrix(c(0, 1, 1, 1, 0, 1, 1, 1, 0), ncol = 3), W = "weka.classifiers.trees.J48", M = TRUE))
preds2 = predict(fit2, test_2)
# accuracy
sum(preds2 == test_2$status_group) / length(preds2)

# probamos Random Forest a ver que tal lo hace
library(randomForest)

rf_model = randomForest(status_group~., data=train)
rf_model

pred_rf = predict(rf_model, newdata = test_2)

sum(pred_rf == test_2$status_group) / length(pred_rf) # accuracy


# modelos
model = J48(status_group~., data=datos, control = Weka_control(M = 2, C = 0.31)) # normal


# Random forest para comparacion
model2 = randomForest(status_group~., data=datos)

# summarize the fit
summary(model)


# make predictions
predictions = predict(model, test)

predictions2 = predict(model2, test)

mean(predictions==predictions2, na.rm = T) # comparamos con otros modelos con diferentes variables


# ahora hago con grid search
# 10 fcv

model_cv = evaluate_Weka_classifier(model, newdata = NULL, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE) #78% en train

# submission
submission = data.frame(id = submissionformat$id, status_group = predictions)

write.csv(submission, file = paste(camino, 'submission_abignu.csv', sep = ''), row.names = F)
write.csv(submission, file = paste(camino, 'submission_rf.csv', sep = ''), row.names = F)
