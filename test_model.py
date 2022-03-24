"""Loading a model generated with create_CNNs.py and testing it with
a test set of images in one folder (with the subfolders named by the 
classes their images represent).""" 
import numpy as np
import json
from os.path import join
from os import listdir, walk
import tensorflow as tf
from tensorflow import keras

def test_model(model_folder, test_img_folder):
    """The function that loads the CNN model and tests its
    functioning with a test image folder.

    Args:
        model_folder (string): The path to the folder that contains the model and 
                                its parameters
        test_img_folder (string): The path to the folder that contains the test images

    Returns:
        float: The model's test accuracy (0...1)
    """

    # Loading the model and the parameters
    model = tf.keras.models.load_model(model_folder)
    with open(join(model_folder, "params.json")) as json_file:
        param_data = json.load(json_file)
        json_file.close()
    class_names = param_data["class_names"]
    img_height = param_data["img_height"]
    img_width = param_data["img_width"]

    # Total number of images
    n_imgs = sum([len(files) for r, d, files in walk(test_img_folder)])

    # Testing with images
    correct_predictions = 0
    dir_list = listdir(test_img_folder)
    for class_folder in dir_list:
        class_path = join(test_img_folder, class_folder)
        file_list = listdir(class_path)
        for img_name in file_list:
            img_path = join(class_path, img_name)
            
            # Ladataan kuva keras-maailmaan
            img = keras.preprocessing.image.load_img(img_path, 
                target_size=(img_height, img_width))

            img_array = keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0) # Create a batch

            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])

            predicted_class = class_names[np.argmax(score)] 

            # If the prediction was correct, we'll increase
            # the count of correct predictions.
            if predicted_class == class_folder:
                correct_predictions += 1

    test_accuracy = correct_predictions / n_imgs
    print(f"Correct predictions: {correct_predictions}")
    print(f"Total number of images: {n_imgs}")    
    print(test_accuracy)
    return test_accuracy

if __name__ == "__main__":
    model_path = r"drive:\Path\to\model"
    test_img_path = r"drive:\Path\to\images"
    test_model(model_path, test_img_path)
     