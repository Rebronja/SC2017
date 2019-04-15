import os
import platform
import warnings
import gen_images
import prep_dataset
import time
import dataset
import ai_zoo

warnings.filterwarnings("ignore")

def display_title_bar():
    # Clears the terminal screen, and displays a title bar.
    clear()

    print("\t***********************************************************")
    print("\t**********************  HiraganaOCR  **********************")
    print("\t***********************************************************")

def menu():
        # Let users know what they can do.
        print("\n[1] Use K Nearest Neighbours algorithm.")
        print("[2] Use Random Forest algorithm.")
        print("[3] Use Stochastic Gradient Descent.")
        print("[4] Use Linear Support Vector Machine.")
        print("[5] Use Simple Neural Network (Smaller images will be used).")
        print("[6] Use Convolutional Neural Network.")
        print("[7] Use Long Short Term Memory Recurrent Neural Network. *WIP*")
        print("[8] Use Multilayered LSTM RNN.")
        print("[9] Use VGG16 (OxfordNet). *WIP*")
        print("[r] Recreate dataset.")
        print("[q] Thats all for now.\n")

        return input("What are we doing today? ")

def clear():
    plat = platform.system()

    if plat == 'Windows':
        os.system('cls')
    else:
        os.system('clear')


clear()

choice = ''
if [f for f in os.listdir('images') if not f.startswith('.')] == []:
    gen_images.generate_images()
if not os.path.isfile('dset.hdf5'):
    prep_dataset.prepare_dataset()
display_title_bar()
dset = dataset.HiraSet('dset', 22500)
dset.pull()
(train_data,test_data,train_labels,test_labels) = dset.require_new(25,20)

while choice.lower() != 'q':
    choice = menu()

    if choice == '1':
        print('\nPlease wait...')
        aiZoo.knn(train_data,test_data,train_labels,test_labels)
        input("All done! Press enter to proceed.")
    elif choice == '2':
        print('\nPlease wait...')
        aiZoo.randomForest(train_data,test_data,train_labels,test_labels)
        input("All done! Press enter to proceed.")
    elif choice == '3':
        print('\nPlease wait...')
        aiZoo.sgd(dset)
        input("All done! Press enter to proceed.")
    elif choice == '4':
        print('\nPlease wait...')
        aiZoo.svm(dset)
        input("All done! Press enter to proceed.")
    elif choice == '5':
        print('\nPlease wait...')
        aiZoo.neuralNet50(dataset)
        input("All done! Press enter to proceed.")
    elif choice == '6':
        print('\nPlease wait...')
        aiZoo.cnn(train_data,test_data,train_labels,test_labels)
        input("All done! Press enter to proceed.")
    elif choice == '7':
        print('\nPlease wait...')
        aiZoo.lstm(train_data,test_data,train_labels,test_labels)
        input("All done! Press enter to proceed.")
    elif choice == '8':
        print('\nPlease wait...')
        aiZoo.multipleLSTM(dset)
        input("All done! Press enter to proceed.")
    elif choice == '9':
        print('\nPlease wait...')
        aiZoo.vgg16(dset)
        input("All done! Press enter to proceed.")
    elif choice.lower() == 'r':
        print('\nPlease wait...')
        gen_images.generate_images()
        if os.path.isfile('dset.hdf5'):
            os.remove('dset.hdf5')
        prep_dataset.prepare_dataset()
        dset = dataset.HiraSet('dset', 22500)
        dset.pull()
        (train_data,test_data,train_labels,test_labels) = dset.require_new(25,20)
        input("All done! Press enter to proceed.")
    elif choice.lower() == 'q':
        print("\nAlrighty. See you!")

    if choice.lower() != 'q':
        display_title_bar()
