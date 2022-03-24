# CNN_tests
Repository for testing different CNN models with image data saved to the computer

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

The must thing to have is 
- [Python 3.X](https://www.python.org/downloads/). The version used in the development was 3.7.0.

### Installing

It is recommed to use virtual environments with the Python projects. You may only use Python in PATH. Then, you do not need the virtual environment. A virtual environment is made like this with a Windows computer:

```
<path_to_python>\python.exe -m venv <name_of_the_environment>
```

The folder with the name <name_of_the_environment> appears. First, clone this repository. Then, make the virtual environment inside the project folder.

If you are using Visual Studio Code, it should detect the virtual environment when opening it inside the project folder. Either by choosing the folder from File -> Open Folder or from command prompt with a command

```
code .
```
if you are in the project folder. The virtual environment name should be seen in the lower left corner after the interpreter name, i.e. Python 3.7.0 64-bit ('virtual_env'). If it does not, it can be selected manually by cliking the interpreter name -> Enter interpreter path... and browsing to the _<name_of_the_environment>\Scripts\python.exe_ When the virtual environment is activated, its name is in bracket in the terminal before the path. Like this:

```
(virtal_env) C:\Users\<my_id>\<my_fancy_folder>\>
```

The virtual enviroment can be activated manually in Visual Studio Code by opening a new terminal window from the + symbol. Command prompt is recommended. PowerShell may have issues with rights.

Next, install the requirements

```
pip install -r requirements.txt
```

Now, everything should be installed (I hope).

### Teaching Different CNN Models

**create_CNNs.py** is the main program for building and teaching the neural network model. The user can choose between seven different architectures: basic CNN, AlexNet, GoogLeNet, VGG, NiN, ResNet and DenseNet. The functions in **create_CNNs.py** call the functions in **modern_CNN_blocks.py** during building of some of the models. All more complex models are built by the aid of the book Dive into Deep Learning (https://d2l.ai).

The user needs to modify the main section at the end of the file **create_CNNs.py**. The path to the image folder where the training data is, the size of the images to be used and the used network should be defined there. The image data folder should contain the subfolders that are named after the class names. Each subfolder should then contain the images that belong to that class. Like this:

<pre>
├── teaching_data_folder
│   ├── name_of_class1
│   |   ├── img1
│   |   ├── img2
|   |   |    ...
│   |   └── imgn
|   |
│   ├── name_of_class2
│   ├── name_of_class3
|   |        ...
│   └── name_of_classm
</pre>

The script will save the checkpoin to the model with the highest validation accuracy during teaching. After the teaching, the weights of the best model, the teaching parameters and the accuracies and losses of each training epoch are saved to a result folder.  

### Testing the Model

**test_model.py** is the file used for testing the created model. The result folder created by the file **create_CNNs.py** and the path to the test images need to be given to that file. The test image folder needs to be in similar format as the teaching data folder. The script **test_model.py** gives the test accuracy (correct predictions/number of test images) as its output.
  
### Testing the Effect of the Size of Teaching Data
  
The file **subset_test.py** includes a script to generate a random subset folder of the teaching data and repeat this kind of subset tests for a number of times. It can be used to test the effect of the size of the teaching data to the model accuracy.
  
  
