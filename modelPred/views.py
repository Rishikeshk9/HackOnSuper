from django.shortcuts import render
import os
import json
import tensorflow as tf
import numpy as np
from django.core.files.storage import default_storage
from django.conf import settings


physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#testingdirect = "F://xampp//htdocs//upload"
testingdirect = "media/upload"



xray = tf.keras.models.load_model(
    'F://Projects//JupyterWorkspace//saved_model//xraymodel', compile=False)

ctscan = tf.keras.models.load_model(
    'F://Projects//JupyterWorkspace//saved_model//ctscanmodel', compile=False)

# Create your views here.
xray.compile(optimizer=tf.keras.optimizers.SGD(lr=0.004),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])
                
ctscan.compile(optimizer=tf.keras.optimizers.SGD(lr=0.004),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

name = ['positive','normal']


def titlename(arr):
    for i in range(1):
        loc = np.where(arr[i] == np.amax(arr[i]))
        x = np.array(loc, dtype=np.int64)
        #print("location: ",loc)
        y = x[0][0]
        return name[y]


def standardizer(pred):
    if (pred[0][0]>pred[0][1]):
        return True
    else:
        return False
    
 

def xraypredict(request):
    if 'uploads.jpg' in os.listdir('media/upload/xray'):
        os.remove('media/upload/xray/uploads.jpg')

    #save_path = os.path.join(settings.MEDIA_ROOT, 'uploads',request.FILES['file'])

    path = default_storage.save("upload/xray/uploads.jpg", request.FILES['file'])

    if 'data_file.json' in os.listdir('media'):
        os.remove('media/data_file.json')

    testing_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255).flow_from_directory(testingdirect,
                                              target_size=(224, 224),
                                              batch_size=1,
                                              classes=['xray'])

    imgs, labels = next(testing_generator)

    pred = xray.predict(imgs)

    #with open("media/data_file.json", "w+") as write_file:
    #    json.dump(response, write_file)

    #a = "'" + cont.replace("'", '"') + "'"

    a = standardizer(pred)

    context = {'resultjson': a}
    return render(request, 'modelPred/predictserver.html', context)



def ctspredict(request):
    if 'uploads.jpg' in os.listdir('media/upload/ctscan'):
        os.remove('media/upload/ctscan/uploads.jpg')

    #save_path = os.path.join(settings.MEDIA_ROOT, 'uploads',request.FILES['file'])

    path = default_storage.save("upload/ctscan/uploads.jpg", request.FILES['file'])

    if 'data_file.json' in os.listdir('media'):
        os.remove('media/data_file.json')

    testing_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255).flow_from_directory(testingdirect,
                                              target_size=(224, 224),
                                              batch_size=1,
                                              classes=['ctscan'])

    imgs, labels = next(testing_generator)

    pred = ctscan.predict(imgs)

    #with open("media/data_file.json", "w+") as write_file:
    #    json.dump(response, write_file)

    #a = "'" + cont.replace("'", '"') + "'"

    a = standardizer(pred)

    context = {'resultjson': a}
    return render(request, 'modelPred/predict2.html', context)
