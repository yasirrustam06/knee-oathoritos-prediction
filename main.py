import cv2,os
import numpy as np
import tensorflow as tf
import matplotlib.cm as cm
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import load_model
from IPython.display import Image, display
from keras.callbacks import ModelCheckpoint
from tensorflow.keras import models, layers
from sklearn.model_selection import train_test_split

print("helo.........")

grades_numbers = ['KL0','KL1','KL2','KL3','KL4']

# label_dict={'0Normal': 0, '1Doubtful': 1, '2Mild': 2, '3Moderate': 3, '4Severe': 4}
class_names = ['Normal','Doubtful','Mild','Moderate','Severe']


model = load_model("knee_models/knee_model3.h5")
# print(model.summary())
image_path = "knee_models/NormalG0 (1).png"


#     image processing steps....
def process_image(image_path):
    Image = cv2.imread(image_path)
    img_gray = cv2.cvtColor(Image,cv2.COLOR_BGR2GRAY)
    res_img = cv2.resize(img_gray,(100,100))
    normalized=res_img/255.0
    Image_pred=np.reshape(normalized,(1,100,100,1))
    return Image_pred



Image_pred = process_image(image_path)
print(Image_pred.shape)


def get_prediction():
    Image_pred = process_image(image_path)
    pred_res = model.predict(Image_pred)
    predictions = []
    for result in pred_res[0]:
        num = float(result)
        num = "{:.5f}".format(num)
        predictions.append(num)
    return predictions






# Create a dictionary to store the labels and their corresponding probabilities
label_probabilities = dict(zip(grades_numbers,get_prediction()))
probabilites = []
# Print the label probabilities
for label, probability in label_probabilities.items():
    PROB = f'{label}: {probability}'
    probabilites.append(PROB)

print(probabilites)


class_probabilities = dict(zip(class_names,get_prediction()))

CLASS_PROBB = []
for label1, probability1 in class_probabilities.items():
    cls_prob = f'{label1}: {probability1}'
    CLASS_PROBB.append(cls_prob)

print(CLASS_PROBB)

title_str = '\n'.join([str(CLASS_PROBB), str(probabilites)])

def make_gradcam_attention_map(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()




preprocess_input = keras.applications.xception.preprocess_input
decode_predictions = keras.applications.xception.decode_predictions
last_conv_layer_name = "conv2d_14"

heatmap = make_gradcam_attention_map(Image_pred, model, last_conv_layer_name)
# plt.imshow(heatmap)






def save_and_display_gradcam(img_path, heatmap, alpha=0.2):
    # Load the original image
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
    plt.figure(figsize=(15,15))


    plt.title(title_str)
    plt.imshow(superimposed_img, cmap='gray')
    plt.show()


save_and_display_gradcam(image_path, heatmap)



