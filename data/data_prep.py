""" Data Preparation

Split the data into training and validation
"""
import glob, cv2, os
from shutil import copyfile
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def split_train_val(input_folder, create_folder, train_folder_name, val_folder_name, test_size=0.1, input_image_format=".png"):
    """
    Given Input folder which contains folders of images with different classes, This will create a folder (create_folder) and divide the dataset into train and val according to the test_size and copy them to train_folder_name and val_folder_name which are inside the create folder.

    Mention input_image_format and test_size if the defaults doesn't work for you.

    """
    if not os.path.exists(create_folder):
        os.makedirs(create_folder)

    files = glob.glob(input_folder+"/*/")
    files = [glob.glob(i+"/*.png") for i in files]

    print("files in each folder", [len(i) for i in files])
    print("Names of the files", [files[i][0].rsplit("/")[-2] for i in range(len(files))])

    for i in tqdm(range(len(files))):
        x = files[i]
        train, test = train_test_split(x, test_size=0.1, random_state = 42)
        train_loc = train[0].rsplit("/")[1]
        train_loc = create_folder+"/"+train_folder_name+"/"+train_loc
        if not os.path.exists(train_loc):
            os.makedirs(train_loc)
        for j in train:
            copyfile(j, train_loc+"/"+j.rsplit("/")[2])
        test_loc = test[0].rsplit("/")[1]
        test_loc = create_folder+"/"+val_folder_name+"/"+test_loc
        if not os.path.exists(test_loc):
            os.makedirs(test_loc)
        for m in test:
            copyfile(m, test_loc+"/"+m.rsplit("/")[2])

    files = glob.glob(create_folder+"/"+train_folder_name+"/*/")
    files = [glob.glob(i+"/*"+input_image_format) for i  in files]
    print("total number of files in train folder: {}".format(sum([len(i) for i in files])))

    files = glob.glob(create_folder+"/"+val_folder_name+"/*/")
    files = [glob.glob(i+"/*"+input_image_format) for i  in files]
    print("total number of files in val folder: {}".format(sum([len(i) for i in files])))


if __name__ == '__main__':
    split_train_val("train", "training_data", "train", "val")
