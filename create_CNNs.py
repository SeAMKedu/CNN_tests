"""Teaching the model using a folder structure where the folder
names are treated as class names. Different CNN types can be used 
(basic CNN, AlexNet, GoogLeNet, VGG, NiN, ResNet, DenseNet)"""
import matplotlib.pyplot as plt
import json
from os.path import join, split
from datetime import datetime
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from modern_cnn_blocks import Inception, ResnetBlock, DenseBlock, TransitionBlock

def divide_to_datasets(img_path, img_shape, val_split, batch_size):
    """Divides the image data to training and validation datasets

    Args:
        img_path (string): Path to the folder containing images with subfolders named after
                           the class names
        img_shape (tuple): (image height, image width, channels) or (image height, image width) 
        val_split (float): The fraction the images used for validation, 0...1
        batch_size (int): Batch size for training images

    Returns:
        tensorflow set, tensorflow set, int: training dataset, validation dataset, number of classes
    """
    img_size = img_shape[:2]
    if len(img_shape) == 3:
        clr_mode = "rgb"
    else:
        clr_mode = "grayscale"

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    img_path,
    validation_split=val_split,
    subset="training",
    seed=123,
    image_size=img_size,
    color_mode=clr_mode,
    batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    img_path,
    validation_split=val_split,
    subset="validation",
    seed=123,
    image_size=img_size,
    color_mode=clr_mode,
    batch_size=batch_size)

    # Get class names before fetching
    class_names = train_ds.class_names

    # Dataset.cache() - kuvat pysyvät muistisssa
    # Dataset.prefetch() - datan esikäsittely ja treenaus tehdään limittäin
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names

def create_alexnet(num_classes):
    """Creates the AlexNet CNN (Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). 
    Imagenet classification with deep convolutional neural networks. Advances in 
    neural information processing systems (pp. 1097–1105)) as instructed in Dive into 
    Deep Learning chapter 7.1 (https://d2l.ai/chapter_convolutional-modern/alexnet.html)

    Args:
        num_classes (int): Number of classes

    Returns:
        tensorflow model: The generated AlexNet model
    """
    model = Sequential([
     # Here, we use a larger 11 x 11 window to capture objects. At the same
        # time, we use a stride of 4 to greatly reduce the height and width of
        # the output. Here, the number of output channels is much larger than
        # that in LeNet
        layers.Conv2D(filters=96, kernel_size=11, strides=4,
                               activation='relu'),
        layers.MaxPool2D(pool_size=3, strides=2),
        # Make the convolution window smaller, set padding to 2 for consistent
        # height and width across the input and output, and increase the
        # number of output channels
        layers.Conv2D(filters=256, kernel_size=5, padding='same',
                               activation='relu'),
        layers.MaxPool2D(pool_size=3, strides=2),
        # Use three successive convolutional layers and a smaller convolution
        # window. Except for the final convolutional layer, the number of
        # output channels is further increased. Pooling layers are not used to
        # reduce the height and width of input after the first two
        # convolutional layers
        layers.Conv2D(filters=384, kernel_size=3, padding='same',
                               activation='relu'),
        layers.Conv2D(filters=384, kernel_size=3, padding='same',
                               activation='relu'),
        layers.Conv2D(filters=256, kernel_size=3, padding='same',
                               activation='relu'),
        layers.MaxPool2D(pool_size=3, strides=2),
        layers.Flatten(),
        # Here, the number of outputs of the fully-connected layer is several
        # times larger than that in LeNet. Use the dropout layer to mitigate
        # overfitting
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes)
    ])
    return model

def vgg_block(num_convs, num_channels):
    """Creates a VGG block consisting of given number of convolution
    layers with given number of channels

    Args:
        num_convs (int): Number of convolution layers in the block
        num_channels (int): Number of channels in the convolution layers

    Returns:
        tensorflow model: Sequential model describing the block
    """
    blk = Sequential()
    for _ in range(num_convs):
        blk.add(
            tf.keras.layers.Conv2D(num_channels, kernel_size=3,
                                   padding='same', activation='relu'))
    blk.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    return blk

def create_vgg(num_classes, conv_arch=((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))):
    """Creates a VGG network (Simonyan, K., & Zisserman, A. (2014). Very deep 
    convolutional networks for large-scale image recognition. arXiv preprint 
    arXiv:1409.1556.) as instructed in Dive into Deep Learning chapter 7.2 
    (https://d2l.ai/chapter_convolutional-modern/vgg.html). 
    The default network is VGG-11.

    Args:
        num_classes (int): Number of classes    
        conv_arch (tuple, optional): The convolution layer architecture. Defaults to 
                                    ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512)).

    Returns:
        tensorflow model: The VGG network
    """
    net = tf.keras.models.Sequential()

    # The convolutional part
    for (num_convs, num_channels) in conv_arch:
        net.add(vgg_block(num_convs, num_channels))
    # The fully-connected part
    net.add(Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes)]))
    return net

def nin_block(num_channels, kernel_size, strides, padding):
    """Creates a NiN block consisting of three convolutional
    layers of which the two latter are 1 x 1. 

    Args:
        num_channels (int): Number of channels
        kernel_size (int): Kernel size (odd)
        strides (int): Stride
        padding (int): Padding

    Returns:
        tensorflow model: Sequential model describing the block
    """
    return Sequential([
        tf.keras.layers.Conv2D(num_channels, kernel_size, strides=strides,
                               padding=padding, activation='relu'),
        tf.keras.layers.Conv2D(num_channels, kernel_size=1,
                               activation='relu'),
        tf.keras.layers.Conv2D(num_channels, kernel_size=1,
                               activation='relu')])

def create_nin(num_classes):
    """Creates a NiN network (Lin, M., Chen, Q., & Yan, S. (2013). Network 
    in network. arXiv preprint arXiv:1312.4400.) as instructed in Dive 
    into Deep Learning chapter 7.3 
    (https://d2l.ai/chapter_convolutional-modern/nin.html). 

    Args:
        num_classes (int): Number of classes

    Returns:
        tensorflow model: The NiN model
    """
    return tf.keras.models.Sequential([
        nin_block(96, kernel_size=11, strides=4, padding='valid'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        nin_block(256, kernel_size=5, strides=1, padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        nin_block(384, kernel_size=3, strides=1, padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        tf.keras.layers.Dropout(0.5),
        nin_block(num_classes, kernel_size=3, strides=1, padding='same'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Reshape((1, 1, num_classes)),
        # Transform the four-dimensional output into two-dimensional output
        # with a shape of (batch size, num_classes)
        tf.keras.layers.Flatten(),])

def googlenet_block1():
    """Creates the first block of GoogLeNet.

    Returns:
        tensorflow model: The model of the first block
    """
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, 7, strides=2, padding='same',
                               activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])

def googlenet_block2():
    """Creates the second block of GoogLeNet.

    Returns:
        tensorflow model: The model of the second block
    """
    return tf.keras.Sequential([
        layers.Conv2D(64, 1, activation='relu'),
        layers.Conv2D(192, 3, padding='same', activation='relu'),
        layers.MaxPool2D(pool_size=3, strides=2, padding='same')])

def googlenet_block3():
    """Creates the third block of GoogLeNet.

    Returns:
        tensorflow model: The model of the third block
    """
    return tf.keras.models.Sequential([
        Inception(64, (96, 128), (16, 32), 32),
        Inception(128, (128, 192), (32, 96), 64),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])

def googlenet_block4():
    """Creates the fourth block of GoogLeNet.

    Returns:
        tensorflow model: The model of the fourth block
    """
    return tf.keras.Sequential([
        Inception(192, (96, 208), (16, 48), 64),
        Inception(160, (112, 224), (24, 64), 64),
        Inception(128, (128, 256), (24, 64), 64),
        Inception(112, (144, 288), (32, 64), 64),
        Inception(256, (160, 320), (32, 128), 128),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])

def googlenet_block5():
    """Creates the fifth block of GoogLeNet.

    Returns:
        tensorflow model: The model of the fourth block
    """
    return tf.keras.Sequential([
        Inception(256, (160, 320), (32, 128), 128),
        Inception(384, (192, 384), (48, 128), 128),
        tf.keras.layers.GlobalAvgPool2D(),
        tf.keras.layers.Flatten()])

def create_googlenet(num_classes):
    """Creates a GoogLeNet network (Szegedy, C., Liu, W., Jia, Y., 
    Sermanet, P., Reed, S., Anguelov, D., … Rabinovich, A. (2015). 
    Going deeper with convolutions. Proceedings of the 
    IEEE conference on computer vision and pattern recognition, 
    pp. 1–9.) as instructed in Dive 
    into Deep Learning chapter 7.4 
    (https://d2l.ai/chapter_convolutional-modern/googlenet.html). 

    Args:
        num_classes (int): Number of classes

    Returns:
        tensorflow model: The NiN model
    """
    return tf.keras.Sequential([
        googlenet_block1(), googlenet_block2(), googlenet_block3(),
        googlenet_block4(), googlenet_block5(), tf.keras.layers.Dense(num_classes)])

def create_resnet(num_classes):
    """Creates Resnet model (He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep 
    residual learning for image recognition. Proceedings of the IEEE conference 
    on computer vision and pattern recognition (pp. 770–778)) as instructed in 
    Dive into Deep Learning Chapter 7.6 
    (https://d2l.ai/chapter_convolutional-modern/resnet.html)

    Args:
        num_classes (int): Number of classes

    Returns:
        tensorflow model: Resnet model
    """
    return tf.keras.Sequential([
        # The first block is like in GoogLeNet except it has a batch 
        # normalization layer
        tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
        
        # Then we have four Resnet blocks and output layer
        ResnetBlock(64, 2, first_block=True),
        ResnetBlock(128, 2),
        ResnetBlock(256, 2),
        ResnetBlock(512, 2),
        tf.keras.layers.GlobalAvgPool2D(),
        tf.keras.layers.Dense(units=num_classes)])

def densenet_block1():
    """The first block of DenseNet"""
    
    return tf.keras.Sequential([
        layers.Conv2D(64, kernel_size=7, strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPool2D(pool_size=3, strides=2, padding='same')])

def densenet_block2():
    """The second block of DenseNet"""

    net = densenet_block1()
    # `num_channels`: the current number of channels
    num_channels, growth_rate = 64, 32
    num_convs_in_dense_blocks = [4, 4, 4, 4]

    for i, num_convs in enumerate(num_convs_in_dense_blocks):
        net.add(DenseBlock(num_convs, growth_rate))
        # This is the number of output channels in the previous dense block
        num_channels += num_convs * growth_rate
        # A transition layer that halves the number of channels is added
        # between the dense blocks
        if i != len(num_convs_in_dense_blocks) - 1:
            num_channels //= 2
            net.add(TransitionBlock(num_channels))
    return net

def create_densenet(num_classes):
    """Creates DenseNet (Huang, G., Liu, Z., Van Der Maaten, L., 
    & Weinberger, K. Q. (2017). Densely connected convolutional 
    networks. Proceedings of the IEEE conference on computer 
    vision and pattern recognition, pp. 4700–4708) as instructed 
    in Dive into Deep Learning Chapter 7.6 
    (https://d2l.ai/chapter_convolutional-modern/densenet.html)

    Args:
        num_classes (int): Number of classes

    Returns:
        tensorflow model: DenseNet model
    """
    net = densenet_block2()
    net.add(layers.BatchNormalization())
    net.add(layers.ReLU())
    net.add(layers.GlobalAvgPool2D())
    net.add(layers.Flatten())
    net.add(layers.Dense(num_classes))
    return net

def create_basic_cnn(num_classes):
    """Creates a basic CNN

    Args:
        num_classes (int): Number of classes

    Returns:
        tensorflow model: The CNN model
    """
    # Bulding the model
    # The size of convolution layers increases because at first the focus is in simple
    # forms and shapes such as lines and features (and raw image data is also noisy) but
    # the deeper we go, the more complex combinations we are willing to form. 
    # https://datascience.stackexchange.com/questions/55545/in-cnn-why-do-we-increase-the-number-of-filters-in-deeper-convolution-layers-fo
    # 3 is a god size to the convolution kernel - Dive into Deep Learning, 7.2.4
    model = Sequential([
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax') # https://datascience.stackexchange.com/questions/22760/number-and-size-of-dense-layers-in-a-cnn
    ])
    return model

def add_normalization_layer(model, img_shape):
    """Adds a normalization layer to the first layer of the network

    Args:
        model (tensorflow model): The current model
        img_shape (tuple): (height, width, channels) or (height, width) 

    Returns:
        tensorflow model: The model with the normalization layer
    """
    
    # Normalization layer: pixel values from 0...255 to  0...1
    norm_model = keras.Sequential(layers.experimental.preprocessing.Rescaling(1./255, input_shape=img_shape))
    norm_model.add(model)

    return norm_model

def add_augmentation_layer(model, img_shape):
    """Adds data augmentation layer to the existing model

    Args:
        model (tensorflow model): The current tensorflow model
        img_shape (tuple): (height, width, channels) or (height, width)

    Returns:
        tensorflow model: The model with added layer
    """
    # Rotating, flipping and zooming the images --> more data is acquired
    data_augmentation = Sequential(
        [
        layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                    input_shape=img_shape),
        layers.experimental.preprocessing.RandomRotation(0.1),
        layers.experimental.preprocessing.RandomZoom(0.1)
        ]
    )
    data_augmentation.add(model)
    
    return data_augmentation

def train_model(model, train_ds, val_ds, epochs, checkpoint_filepath):
    """Sets compilation parameters and trains the model

    Args:
        model (tensorflow model): The model to be trained
        train_ds (tensorflow set): Training dataset
        val_ds (tensorflow set): Validation dataset
        epochs (int): Number of epochs
        checkpoint_filepath (sring): The path to save the checkpoint

    Returns:
        tensorflow model, model history: trained model, model's training history
    """
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    # Checkpoint to the weights of the best model
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    # Model training starts
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[model_checkpoint_callback]
        )
    
    return model, history 

def save_model(model, history, checkpoint_filepath, img_path, img_shape, class_names,
                save_base_folder=None, extra_tag=None):
    """Saves the model and the teaching parameters into a folder named
    by the time and accuracy in the subfolder 'models' in the image 
    folder.

    Args:
        model (tensorflow model): The model to be saved
        history (tensorflow history): The model history
        checkpoint_filepath (string): The filepath to the checkpoint containing the best weights
        img_path (string): The image folder
        img_shape (tuple): (height, width, channels) or (height, width)
        class_names (list): List of class names
        save_base_folder (string, optional): The folder where the new folder for the saved 
                                            model and its parameters is generated. If None, 
                                            img_path is used. Defaults to None.
        extra_tag (string, optional): A tag that is added to the name of the folder 
                                            the model and the parameters are saved to. 
                                            Defaults to None.
    
    Returns:
        string: the folder the model was saved to
    """
    # Current time
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Loading the weights that gave the highest validation accuracy from the checkpoint
    model.load_weights(checkpoint_filepath)

    # Accuracies (based on the best validation accuracy since we are using the weights that
    # resulted it)
    val_acc = max(history.history['val_accuracy'])
    ind = history.history['val_accuracy'].index(val_acc)
    acc = history.history['accuracy'][ind]

    # Saving the model
    if save_base_folder is None:
        save_base_folder = split(img_path)[0]
    mdl_name = f"{dt_string}_acc_{acc:.4f}_val_acc_{val_acc:.4f}"
    if extra_tag is not None:
        mdl_name = mdl_name + " " + extra_tag
    model_folder = join(save_base_folder, "models")
    model_folder = join(model_folder, mdl_name)
    model.save(model_folder)

    # Saving the model parameters
    param_data = json.dumps({"class_names": class_names,
                            "img_height": img_shape[0],
                            "img_width": img_shape[1]}, indent=4)
    with open(join(model_folder, "params.json"), "w") as savefile:
        savefile.write(param_data)
        savefile.close()
    
    return model_folder


def visualize_results(history, epochs, save_folder, show_results=True):
    """Draws training and validation accuracies and losses
    over epochs.

    Args:
        history (tensorflow history): Model training history
        epochs (int): Number of epochs
        save_folder (string): Folder to save the graph to (if None, the figure is not saved)
        show_results (bool, optional): Is the graph shown on the screen or not. 
                                        Defaults to True.
    """
    # Visualizing the results
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    # Saving the figure and its data
    if save_folder is not None:
        plt.savefig(join(save_folder, "results.png"))
        plot_data = json.dumps({"epochs": epochs,
                "acc": acc,
                "val_acc": val_acc,
                "loss": loss,
                "val_loss": val_loss}, indent=4)
        with open(join(save_folder, "results.json"), "w") as savefile:
            savefile.write(plot_data)
            savefile.close()
    if show_results:
        plt.show()

def main(cnn_type, image_path, image_shape, batch_size, epochs=200,
        save_results=True, save_base_folder=None, save_extra_tag=None, show_graphs=True):    
    """The main function

    Args:
        cnn_type (string): The CNN architecture:
                    "basic", "AlexNet", "VGG", "NiN", "GoogLeNet", "ResNet" or "DenseNet"
        image_path (string): The path to the folder containing the subfolders named by the 
                                class names that their images represent
        image_shape (tuple): (image height, image width, channels) or 
                                (image height, image width)
        batch_size (int): The size of a training batch
        epochs (int, optional): Number of training epochs. Defaults to 200.
        save_results (bool, optional): Are the results saved or not. Defaults to True.
        save_base_folder (string, optional): The folder where the new folder for the saved 
                                            model and its parameters is generated. If None, 
                                            img_path is used. Defaults to None.
        save_extra_tag (string, optional): A tag that is added to the name of the folder 
                                            the model and the parameters are saved to. 
                                            Defaults to None.
        show_graphs (bool, optional): Are the graphs shown or not. Defaults to True.

    Raises:
        ValueError: If the parameter cnn_type gets illegal value
    """
    val_split = 0.2
    # Parametrit: batch_size kuvaa, joiden koko on 180 x 180
    #batch_size = 16  # Montako kuvaa harjoituskansiosta käytetään yhdellä myötävirta - 
                    # vastavirta-kierroksella
                    # (Kuinka asettaa batch_size: https://stats.stackexchange.com/questions/164876/what-is-the-trade-off-between-batch-size-and-number-of-iterations-to-train-a-neu) 
    #epochs = 200 # Yksi epookki = yksi myötävirta - vastavirta-kierros KAIKILLA kuvilla
                # (iteraatioita tulee siten kuvien_määrä/batchin_koko per epookki)
    augment_data = True
    checkpoint_filepath = "C:/temp/tf_checkpoint"

    training_ds, validation_ds, class_names = divide_to_datasets(image_path, image_shape, val_split, batch_size)
    n_classes = len(class_names) # Luokkien määrä ulostulokerrosta varten
    if cnn_type == "AlexNet":
        model = create_alexnet(n_classes)
    elif cnn_type == "VGG":
        model = create_vgg(n_classes)
    elif cnn_type == "NiN":
        model = create_nin(n_classes)
    elif cnn_type == "GoogLeNet":
        model = create_googlenet(n_classes)
    elif cnn_type == "ResNet":
        model = create_resnet(n_classes)
    elif cnn_type == "DenseNet":
        model = create_densenet(n_classes)
    elif cnn_type == "basic":
        model = create_basic_cnn(n_classes)
    else:
        raise ValueError(f"Unknown network type: {cnn_type}!")
    model = add_normalization_layer(model, image_shape)
    if augment_data:
        model = add_augmentation_layer(model, image_shape)
    model, history = train_model(model, training_ds, validation_ds, epochs, checkpoint_filepath)
    if save_results:
        save_folder = save_model(model, history, checkpoint_filepath, 
                                image_path, image_shape, class_names, save_base_folder,
                                save_extra_tag)
    else:
         save_folder = None
    visualize_results(history, epochs, save_folder, show_graphs)

if __name__ == "__main__":
    # The path to the train set that contains the data belonging to different 
    # classes in its subfolders. The subfolder names = class names.
    image_path = r"drive:\Path\to\images\train_set" 
    
    # Call to the main function to create the CNN. 
    main("basic", image_path, (512, 512), 512, show_graphs=False)