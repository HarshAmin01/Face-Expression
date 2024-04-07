# importing libraries
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import base64
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Flatten, Dense, Lambda, Reshape, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend as K
import warnings
import tensorflow as tf
warnings.filterwarnings('ignore')
tf.config.run_functions_eagerly(True)

def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim)) 
    return z_mean + K.exp(0.5 * z_log_var) * epsilon 


def vae_encoder(image_size, latent_dim):
    
    inputs = Input(shape=(128,128,3), name='encoder_input')
    x = Conv2D(32, kernel_size=3, strides=2, activation='relu', padding='same')(inputs) 
    x = Conv2D(64, kernel_size=3, strides=2, activation='relu', padding='same')(x) 
    x = Conv2D(64, kernel_size=3, strides=1, activation='relu', padding='same')(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)  
    z_log_var = Dense(latent_dim, name='z_log_var')(x) 
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])    
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')    
    return encoder


def vae_decoder(latent_dim, output_shape):
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')  
    x = Dense(64 * (output_shape[0] // 4) * (output_shape[1] // 4), activation='relu')(latent_inputs)
    x = Reshape((output_shape[0] // 4, output_shape[1] // 4, 64))(x)   
    x = Conv2DTranspose(64, kernel_size=3, strides=2, activation='relu', padding='same')(x) 
    x = Conv2DTranspose(32, kernel_size=3, strides=2, activation='relu', padding='same')(x)
    outputs = Conv2DTranspose(3, kernel_size=3, activation='sigmoid', padding='same')(x) 
    decoder = Model(latent_inputs, outputs, name='decoder') 
    return decoder

input_shape = (128, 128, 3)
inputs = Input(shape=input_shape, name='encoder_input')

latent_dim = 64
encoder = vae_encoder(input_shape, latent_dim)
decoder = vae_decoder(latent_dim, input_shape)

outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')
vae.compile(optimizer='adam', loss='mse', metrics=['accuracy']) 
vae.load_weights("models/VAE.h5")

# flask app
app = Flask(__name__)
CORS(app)


# predictive models
model_paths = {
    "VGGNET19": "models/VGGNET19_Model.h5",
    "Efficient Unfreeze": "models/EfficientNET.h5",
    "Convolution Neural Network": "models/CNN.h5"
}

# loading models
models = {model_name: load_model(model_path) for model_name, model_path in model_paths.items()}
labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

# predictive models
def classify_emotion(image, model_name):
    model = models[model_name]
    image = cv2.resize(image, (128, 128))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = image.astype("float32") / 255.0
    prediction = model.predict(image)
    index = np.argmax(prediction)
    return labels[index]

# generative model
def VAEModel(image):
    image = cv2.resize(image, (128, 128))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = image.astype("float32") / 255.0
    gen = vae.predict(image)
    gen_bytes = io.BytesIO()
    gen_img = Image.fromarray((gen[0] * 255).astype(np.uint8))
    gen_img.save(gen_bytes, format='PNG')
    gen_bytes.seek(0)
    gen_base64 = base64.b64encode(gen_bytes.read()).decode('utf-8')
    return gen_base64

# creating flask API
@app.route('/classify', methods=['POST'])
def classify():
    image_file = request.files['image']
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    model_name = request.form['model']

    _, buffer = cv2.imencode('.jpg', image)
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    
    if model_name == 'VAE':
        gen_image = VAEModel(image)
        return jsonify({'model_name': model_name, 'image': encoded_image, 'gen_image': gen_image})
    else:
        emotion = classify_emotion(image, model_name)
        return jsonify({'emotion': emotion, 'model_name': model_name, 'image': encoded_image})

# running flask app
if __name__ == '__main__':
    app.run(debug=True)
