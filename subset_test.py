from create_CNNs import main
from os import listdir, makedirs, mkdir
from os.path import join, exists
from random import sample
from shutil import copyfile, rmtree
from math import log

def create_subset(img_folder, subset_size):
    """Creating a random subset of a given size from images in a 
    given folder and saving the subset to a new folder.

    Args:
        img_folder (string): Path to the image folder that contains a subfolder
                            "train_set". The subset is made from the "train_set" 
                            subfolder.
        subset_size (int): Subset size (number of images)

    Returns:
        string: The path to the subset (it is created to img_folder/subsets/subset_[subset_size])
    """

    # Naming the subset folder based on its size
    subset_path = join(img_folder, "subsets", f"subset_{subset_size}")
    if exists(subset_path):
        rmtree(subset_path)
    makedirs(subset_path)

    # Each original folder should contain "train_set" and "test_set"
    # We do not need test set
    set_folder = join(img_folder, "train_set")

    # class folders
    class_folder_names = listdir(set_folder)

    # Going through the subfolders one by one and making random subset
    # of each
    for class_folder in class_folder_names:
        class_folder_path = join(set_folder, class_folder)
        files = listdir(class_folder_path)
        random_subset = sample(files, k=subset_size)

        # Going through the files and copying them to new destination 
        for filename in random_subset:
            subset_file_path = join(class_folder_path, filename)
            
            new_class_path = join(subset_path, class_folder)
            if not exists(new_class_path):
                mkdir(new_class_path)

            new_subset_file_path = join(new_class_path, filename)            
            copyfile(subset_file_path, new_subset_file_path)
    return subset_path

if __name__ == "__main__":

    net_type = "VGG"
    data_path = r"drive:\Path\to\images" 
    n_iterations = 1 # Number of iterations for each subset size
    subset_sizes = [4000] # List of subset sizes that are tested
    img_shape = (150, 150, 3) # Image size (width, height, channels)
    desired_n_steps = 20 # Steps per epoch

    # Going through all subset sizes that should be tested
    for subset_size in subset_sizes:

        # Choosing a batch size that yields the desired number of steps per epoch
        x = log(subset_size/desired_n_steps) / log(2)
        batch_size = 2**round(x)

        # Repeating for the iterations
        for i in range(n_iterations):
            subset_path = create_subset(data_path, subset_size) 
            main(net_type, subset_path, img_shape, batch_size, save_base_folder=data_path,
                save_extra_tag=f"subset_{subset_size}_type_{type}", show_graphs=False)
            rmtree(subset_path)