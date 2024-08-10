# Train-Station
A tool for producing .h5 and .keras AI models for image classification for up to 6 classes. Epochs and Batch Size are customizable. Each class must be in the form of a folder with images inside.


# Download Executable
[Dropbox Download](https://www.dropbox.com/scl/fi/d5kf4xjgjkjvjbat25pwz/TrainStation.exe?rlkey=1zrwpd5o5rsnmy9chjvlluo0r&st=ilos91p9&dl=0)


# Usage
1. Download and run executable from Dropbox
2. Click "Add Samples" for each class, and select a folder with only images in it
3. Enter Epochs and Batch Size, and click "Train" **one time**
4. In the terminal that opened with the program, you can watch the training process
5. After the model has been trained, two file paths are shown in the terminal. These are the .h5 and .keras models

# Notes
1. This program will use all resources available to it while training, so manually limiting CPU usage of the program may work best for you.
2. A labels.txt file is available in the folder where the models were saved to



```
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("<MODEL_PATH (.h5)>", compile=False)

# Load the labels
class_names = open("<LABEL_PATH (.txt)>", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
data = np.ndarray(shape=(1, 180, 180, 3), dtype=np.float32)

# Replace this with the path to your image
image = Image.open("<IMAGE_PATH>").convert("RGB")

# resizing the image to be at least 180x180 and then cropping from the center
size = (180, 180)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# turn the image into a numpy array
image_array = np.asarray(image)

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
data[0] = normalized_image_array

# Predicts the model
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

# Print prediction and confidence score
print("Class:", class_name[2:], end="")
print("Confidence Score:", confidence_score)
```
