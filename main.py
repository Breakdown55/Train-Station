import numpy as np
from customtkinter import *
import pathlib
from tkinter import filedialog
import tkinter as tk
import os
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow.keras import layers, models
from keras.models import load_model
import sys
print("Train icon from https://commons.wikimedia.org/wiki/File:TrainClipart.svg\n\n")

def resource_path(relative_path):
    try:
        base_path = pathlib.Path(sys._MEIPASS)
    except Exception:
        base_path = pathlib.Path(__file__).parent.resolve()
    return base_path / relative_path

file = str(resource_path(".")) + "/"

np.set_printoptions(suppress=True)

def load_and_preprocess_image(image_path, img_height=180, img_width=180):
    image = Image.open(image_path).convert('RGB')
    image = ImageOps.fit(image, (img_height, img_width), Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = image_array.astype(np.float32) / 255.0  # Normalize to [0, 1]
    return normalized_image_array


def load_model_and_classes():
    global model, class_names
    model = load_model(resource_path("output.h5"), compile=False)
    class_names = open(resource_path("labels.txt"), "r").readlines()


load_model_and_classes()
data = np.ndarray(shape=(1, 180, 180, 3), dtype=np.float32)


app = CTk()
set_appearance_mode('dark')
app.geometry("800x600")
app.title("Train Station")
app.iconbitmap(resource_path("thumb.ico"))
app.wm_resizable(False, False)

# vars to hold directory paths
class_dirs = [False] * 6


terminal_font = CTkFont(family="Terminal", size=20)
labelsToWrite = []


def class_prompt(index):
    class_dirs[index] = filedialog.askdirectory(title="Select a folder")
    if class_dirs[index]:
        folder_name = os.path.basename(class_dirs[index])
        class_filenames[index].configure(text=folder_name)
        class_titles[index].insert(0, folder_name)  # Set the class title entry to the folder name
        labelsToWrite.append(folder_name)



class_labels = []
class_titles = []
class_add_buttons = []
class_filenames = []

for i in range(6):
    class_labels.append(CTkLabel(master=app, font=terminal_font, text=f"Class {i+1}:"))
    class_titles.append(CTkEntry(master=app, width=200, height=30, placeholder_text=f"Class {i+1}"))
    class_add_buttons.append(CTkButton(master=app, width=200, height=50, text="Add Samples", command=lambda i=i: class_prompt(i)))
    class_filenames.append(CTkLabel(master=app, text=""))


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            preprocessed_image = load_and_preprocess_image(file_path)
            images.append(preprocessed_image)
    return images


def start_train():
    global class_dirs

    # Write class labels to file
    with open(resource_path("labels.txt"), "w") as f:
        for item in labelsToWrite:
            f.write(f"{labelsToWrite.index(item)} {item}\n")

    folder_paths = []
    class_labels_texts = []


    batch_size = int(batchSizeEntry.get())
    epochs = int(epochEntry.get())
    img_height = 180
    img_width = 180

    for i in range(6):
        if class_titles[i].get() != "":
            folder_paths.append(class_dirs[i])
            class_labels_texts.append(class_titles[i].get())

    if len(folder_paths) < 2:
        print("At least two classes are required for training.")
        return

    label_to_index = {class_label: i for i, class_label in enumerate(class_labels_texts)}

    images = []
    labels = []

    for folder_path in folder_paths:
        class_label = os.path.basename(folder_path)
        class_index = label_to_index[class_label]
        class_images = load_images_from_folder(folder_path)
        images.extend(class_images)
        labels.extend([class_index] * len(class_images))

    images_preprocessed = np.array(images)
    labels_array = np.array(labels)

    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),  # Add L2 regularization
        layers.Dropout(0.5),
        layers.Dense(len(class_labels_texts), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(images_preprocessed, labels_array, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    val_loss, val_accuracy = model.evaluate(images_preprocessed, labels_array, verbose=2)
    print(f'Validation accuracy: {val_accuracy}')
    
    model.save(resource_path("output.h5"))
    model.save(resource_path("output.keras"))
    print("\nOutputs At:"+resource_path("output.h5"))
    print(resource_path("output.keras"))
    # reload model post save
    load_model_and_classes()


lineIMG = Image.open(resource_path("images/line.png"))
lineImage = CTkImage(lineIMG, size=(800, 20))
lineImageLbl = CTkLabel(app, text="", image=lineImage)

epochEntry = CTkEntry(master=app, width=100, height=30, placeholder_text="Epochs")
batchSizeEntry = CTkEntry(master=app, width=100, height=30, placeholder_text="Batch Size")
trainButton = CTkButton(master=app, width=200, height=80, font=terminal_font, text="Train", fg_color='#027519', hover_color='#14a330', command=start_train)

epochEntry.place(anchor='center', relx=0.15, rely=0.8)
batchSizeEntry.place(anchor='center', relx=0.15, rely=0.9)
trainButton.place(anchor='center', relx=0.4, rely=0.85)
lineImageLbl.place(anchor='center', relx=0.5, rely=0.7)

shiftx = 0.55
shifty = 0.225
# place ui elements
for i in range(6):
    if i < 3:
        class_labels[i].place(anchor='center', relx=0.075, rely=0.05 + shifty * i)
        class_titles[i].place(anchor='center', relx=0.26, rely=0.05 + shifty * i)
        class_add_buttons[i].place(anchor='center', relx=0.26, rely=0.125 + shifty * i)
        class_filenames[i].place(anchor='center', relx=0.26, rely=0.19 + shifty * i)
    else:
        class_labels[i].place(anchor='center', relx=0.075 + shiftx, rely=0.05 + shifty * (i - 3))
        class_titles[i].place(anchor='center', relx=0.26 + shiftx, rely=0.05 + shifty * (i - 3))
        class_add_buttons[i].place(anchor='center', relx=0.26 + shiftx, rely=0.125 + shifty * (i - 3))
        class_filenames[i].place(anchor='center', relx=0.26 + shiftx, rely=0.19 + shifty * (i - 3))

def open_file():
    file = filedialog.askopenfile(mode='r')
    if file:
        filepath = os.path.abspath(file.name)
        print(filepath)

        previewIMG = Image.open(filepath)
        previewImage = CTkImage(previewIMG, size=(130, 130))

        image_preview = CTkLabel(app, image=previewImage, text="")
        image_preview.place(anchor='center', relx=0.9, rely=0.85)

        preprocessed_image = load_and_preprocess_image(filepath, 180, 180)
        data[0] = preprocessed_image

        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        print("Class:", class_name.strip(), end="")  # remove whitespace
        print("Confidence Score:", confidence_score)

        predictionLabel.configure(text=class_name.strip() + " " + str(confidence_score))

predictionLabel = CTkLabel(app, text="")
predictionLabel.place(anchor='center', relx=0.7, rely=0.9)

previewButton = CTkButton(master=app, command=open_file, text="Preview Model", width=100, height=50, font=terminal_font)
previewButton.place(anchor='center', relx=0.7, rely=0.8)

app.mainloop()
