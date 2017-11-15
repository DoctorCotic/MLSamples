import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input,decode_predictions
import numpy as np

model = VGG16(weights='imagenet')
# грузим предварительно обученные веса

img = image.load_img('picture_test/caty.jpeg', target_size=(224,224))
x = image.img_to_array(img)
#размерность
x = np.expand_dims(x, axis=0)
# предварительная обработка изображений
x = preprocess_input(x)

preds = model.predict(x)
print('Image Recognition Results',decode_predictions(preds,top = 5)[0])