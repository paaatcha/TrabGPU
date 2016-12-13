import os
import glob
import cv2
import datetime
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.optimizers import SGD, Adagrad
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn.metrics import log_loss
from keras import __version__ as keras_version


#funcao para ler as imagens do caminho especi, e redimensiona-la
def get_im_cv2(path):
    img = cv2.imread(path)
    resized = cv2.resize(img, (48, 48), interpolation=cv2.INTER_LINEAR)
    return resized

#carrega os dados de treino
def load_train():
    X_train = []
    X_train_id = []
    y_train = []
    start_time = time.time()

    print('Read train images')
    folders = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    for fld in folders:
        index = folders.index(fld)
        print('Load folder {} (Index: {})'.format(fld, index))
	# caminho da base de treino, deve ser alterado para o caminho que colocar o codigo
        path = os.path.join('..','baseteste2','train', fld, '*.jpg') 
        files = glob.glob(path)
        for fl in files:
            flbase = os.path.basename(fl)
            img = get_im_cv2(fl)
            X_train.append(img)
            X_train_id.append(flbase)
            y_train.append(index)

    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    #retorna os dados, os rotulos e as ids das imagens
    return X_train, y_train, X_train_id  


def load_pre_test():
    X_train = []
    X_train_id = []
    y_train = []
    start_time = time.time()

    print('Read test images')
    folders = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    for fld in folders:
        index = folders.index(fld)
        print('Load folder {} (Index: {})'.format(fld, index))
	# caminho da base de validacao, deve ser alterado para o caminho que colocar o codigo
        path = os.path.join('..','baseteste2','teste', fld, '*.jpg') 
        files = glob.glob(path)
        for fl in files:
            flbase = os.path.basename(fl)
            img = get_im_cv2(fl)
            X_train.append(img)
            X_train_id.append(flbase)
            y_train.append(index)

    print('Read pre test data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_train, y_train, X_train_id

#carrega os dados de teste para submissao
def load_test():
    #caminho da base
    path = os.path.join('..','test_stg1', '*.jpg') 
    files = sorted(glob.glob(path))

    X_test = []
    X_test_id = []
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im_cv2(fl)
        X_test.append(img)
        X_test_id.append(flbase)

    return X_test, X_test_id

#gera o arquivo de submissao
def create_submission(predictions, test_id, info): 
    #probabilidades de cada classe em todos os exemplos de teste,ids da imagens, string
    result1 = pd.DataFrame(predictions, columns=['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
    result1.loc[:, 'image'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    sub_file = 'submission_' + info + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    result1.to_csv(sub_file, index=False)

#normaliza os dados entre 0 e 1
def read_and_normalize_train_data(train=True):
    if train:
       train_data, train_target, train_id = load_train() 
    else:
       train_data, train_target, train_id = load_pre_test()

    print('Convert to numpy...')
    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8) 
    train_data = train_data.transpose((0, 3, 1, 2))

    print('Convert to float...')
    train_data = train_data.astype('float32')
    train_data = train_data / 255
    train_target = np_utils.to_categorical(train_target, 8)

    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    #retorna dados, rotulos e ids das imagens
    return train_data, train_target, train_id 

#normaliza as imagens de teste (base de submissao)
def read_and_normalize_test_data():
    start_time = time.time()
    test_data, test_id = load_test()

    test_data = np.array(test_data, dtype=np.uint8)
    test_data = test_data.transpose((0, 3, 1, 2))

    test_data = test_data.astype('float32')
    test_data = test_data / 255

    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    print('Read and process test data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return test_data, test_id 


def dict_to_list(d):
    ret = []
    for i in d.items():
        ret.append(i[1])
    return ret


def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()


def create_model():
    model = Sequential() 
    #adiciona camada de convolucao com 4 filtros 3x3, seguida por uma funcao de ativacao relu
    model.add(Convolution2D(4, 3, 3, input_shape=(3, 48, 48),activation='relu', dim_ordering='th',init='he_uniform'))
    #adiciona camada de convolucao com 4 filtros 3x3, seguida por uma funcao de ativacao relu
    model.add(Convolution2D(4, 3, 3, activation='relu', dim_ordering='th',init='he_uniform'))
    #adiciona camada de max_pooling, com deslocamento de 2
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
    
    #adiciona camada de convolucao com 8 filtros 3x3, seguida por uma funcao de ativacao relu
    model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='th',init='he_uniform'))
    #adiciona camada de convolucao com 8 filtros 3x3, seguida por uma funcao de ativacao relu
    model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='th',init='he_uniform'))
    #adiciona camada de max_pooling, com deslocamento de 2
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))

    #transforma todos os features maps da camada anterior em um unico vetor
    model.add(Flatten()) 
    #adiciona uma camada densamente conectada com 96 neuronios, seguida por uma funcao de ativacao relu
    model.add(Dense(96, activation='relu',init='he_uniform'))
    #adiciona uma camada densamente conectada com 96 neuronios, seguida por uma funcao de ativacao relu
    model.add(Dropout(0.5))#adiciona dropout com propabilidade 0.5
    #adiciona uma camada densamente conectada com 16 neuronios, seguida por uma funcao de ativacao relu
    model.add(Dense(16, activation='relu',init='he_uniform'))
    model.add(Dropout(0.5))#adiciona dropout com propabilidade 0.5
    #adiciona uma camada densamente conectada com 8 neuronios, seguida por uma funcao de ativacao softmax
    model.add(Dense(8, activation='softmax'))

    #define o treinamento da rede
    sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.8, nesterov=True)
    #adiciona treinamento ao modelo, e define a funcao de avaliacao do erro
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    return model


def get_validation_predictions(train_data, predictions_valid):
    pv = []
    for i in range(len(train_data)):
        pv.append(predictions_valid[i])
    return pv

#treinamento da rede com croos_validation
def run_cross_validation_create_models(nfolds=10):
    batch_size = 24 #tamanho do batch
    nb_epoch = 100   #numeros de epocas
    random_state = 51 
    first_rl = 96 


    train_data, train_target, train_id = read_and_normalize_train_data() #carrega os dados de treino

    yfull_train = dict()
    kf = KFold(len(train_id), n_folds=nfolds, shuffle=True, random_state=random_state) #divide os dados em folds
    num_fold = 0
    sum_score = 0
    models = []   #lista de modelos de rede
    for train_index, test_index in kf:
        model = create_model()           #gera um modelo da rede para cada fold
        X_train = train_data[train_index]   #separa os folds para treino
        Y_train = train_target[train_index]
        X_valid = train_data[test_index]    #separa o fold para validar
        Y_valid = train_target[test_index]

        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_valid), len(Y_valid))

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, verbose=0),
        ]
        model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, #treina o modolo da rede
              shuffle=True, verbose=2, validation_data=(X_valid, Y_valid),
              callbacks=callbacks)

        predictions_valid = model.predict(X_valid.astype('float32'), batch_size=batch_size, verbose=2) #gera a resposta para os dados de validacao
        score = log_loss(Y_valid, predictions_valid)  #calcula o erro na validacao
        print('Score log_loss: ', score)
        sum_score += score*len(test_index)

        # Store valid predictions
        for i in range(len(test_index)):
            yfull_train[test_index[i]] = predictions_valid[i]

        models.append(model) #adiciona modelo a lista

    score = sum_score/len(train_data)
    print("Log_loss train independent avg: ", score)

    info_string = 'loss_' + str(score) + '_folds_' + str(nfolds) + '_ep_' + str(nb_epoch) + '_fl_' + str(first_rl)
    return info_string, models # retorna os modelos treinados

#testa dados separados para testar e gera a resposta dos dados de submissao
def run_cross_validation_process_test(info_string, models):
    batch_size = 24
    num_fold = 0
    pre_test=[]
    yfull_test = []
    test_id = []
    nfolds = len(models)
    
    #testa os dados reservados para teste
    x_teste, y_teste,id_teste = read_and_normalize_train_data(train=False) #carrega os dados para testar
    num_fold=0
    soma=0
    for i in range(nfolds):
        model = models[i]  #carrega um modelo da rede de cada vez
        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        predict = model.predict(x_teste, batch_size=batch_size, verbose=2) #gera a resposta de cada modelo para o teste
        score = log_loss(y_teste, predict) #calcula o erro do modelo
        soma=soma+score
        print('Score log_loss: ', score)
        pre_test.append(predict) #adiciona a resposta a uma lista
    resposta = merge_several_folds_mean(pre_test, nfolds) #calcula a media da resposta de todos os modelos
    print("media ",soma/nfolds)

    acertos=0.0
    print("predito/correto")
    #calcula a acuracia da rede
    for i in range(len(y_teste)):
       print(np.argmax(resposta[i]),np.argmax(y_teste[i])) 
       if(np.argmax(resposta[i])==np.argmax(y_teste[i])): #verifica se a reposta com maior propabilidade e a resposta certa 
           acertos=acertos+1
    print('acertos: ',acertos)
    acuracia=acertos/150
    print('acuracia: ',acuracia)
    
    num_fold=0
    
    #gera as reposta dos dados da base para submissao
    for i in range(nfolds):
        model = models[i]
        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        test_data, test_id = read_and_normalize_test_data()
        test_prediction = model.predict(test_data, batch_size=batch_size, verbose=2)
        yfull_test.append(test_prediction)
    
    test_res = merge_several_folds_mean(yfull_test, nfolds) #calcula a media da resposta dos modelos
    info_string = 'loss_' + info_string \
                + '_folds_' + str(nfolds)
    create_submission(test_res, test_id, info_string) #gera o arquivo de submissao


if __name__ == '__main__':
    print('Keras version: {}'.format(keras_version))
    num_folds = 5 # define o numero de folders
    info_string, models = run_cross_validation_create_models(num_folds)
    run_cross_validation_process_test(info_string, models)
    
