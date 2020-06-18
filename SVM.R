# SVM con OVA
require(tidyverse)
require(liquidSVM)

camino = 'C:/Users/agustin/Google Drive/Master AI/Asignaturas/Mineria de Datos Preprocesamiento y Clasificacion/TrabajoFinal/'

datos = read.csv(paste(camino,'clean.csv', sep = ''), header = T, sep = ',')
test = read.csv(paste(camino,'clean_test.csv', sep = ''), header = T, sep = ',')
submissionformat = read.csv(paste(camino,'SubmissionFormat.csv', sep = ''), header = T, sep = ',')

datos$X = NULL # eliminamos esta columna
id_train = datos$id
datos$id = NULL
datos$date_recorded = NULL # para random forest
datos$subvillage = NULL # lo saco por razones de que disminuye la prediccion

test$X = NULL
id_test = test$id
test$id = NULL
test$date_recorded = NULL # para random forest
test$subvillage = NULL


smp_size <- floor(0.75 * nrow(datos))

#set.seed(123)
train_ind <- sample(seq_len(nrow(datos)), size = smp_size)

train = datos[train_ind,]
test_2 = datos[-train_ind,]

str(datos)
# model
model_SVM = mcSVM(status_group~., train, mc_type="OvA_ls")

preds = predict(model_SVM, test_2)

mean(preds == test_2$status_group)

# model dataset completo
model_SVM = mcSVM(status_group~., datos, mc_type="OvA_ls")

preds = predict(model_SVM, test)

# submission
submission = data.frame(id = submissionformat$id, status_group = preds)

write.csv(submission, file = paste(camino, 'submission_svm.csv', sep = ''), row.names = F)

