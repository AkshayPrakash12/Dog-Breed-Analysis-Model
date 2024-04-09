from keras.applications import VGG16
from keras.preprocessing import image as image_utils
from keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.applications.vgg16 import decode_predictions
import os
import google.generativeai as genai
from tkinter import *
from tkinter import filedialog
import numpy as np

model = VGG16(weights="imagenet")
genai.configure(api_key="{API Key Here]")

# Set up the model
generation_config = {
  "temperature": 0.9,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 2048,
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
]

genaimodel = genai.GenerativeModel(model_name="gemini-1.0-pro",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

convo = genaimodel.start_chat(history=[
])

def show_image(image_path):
    image = mpimg.imread(image_path)
    plt.imshow(image)


def load_and_process_image(image_path):
    image = image_utils.load_img(image_path, target_size=(224, 224))
    image = image_utils.img_to_array(image)
    image = image.reshape(1, 224, 224, 3)
    image = preprocess_input(image)
    return image


def readable_prediction(image_path):
    # Show image
    show_image(image_path)
    # Load and pre-process image
    image = load_and_process_image(image_path)
    # Make predictions
    predictions = model.predict(image)
    if 151 <= np.argmax(predictions) <= 268:
        predicted_breed_list = decode_predictions(predictions, top=1)
        predicted_breed = str(predicted_breed_list)
        predicted_breed = predicted_breed.split(",")
        predicted_breed = predicted_breed[1]
        predicted_breed = str(predicted_breed)
        predicted_breed = predicted_breed.replace("_", " ")
        predicted_breed = predicted_breed.replace("'", "")
        print("The Predicted Breed is the:", predicted_breed)
        global output
        output = predicted_breed
        output = str(output)
    else:
        tkinterReadableoutput.set("Not a dog!")
        exit(0)
    # Print predictions in readable form





# prints all files

def openTextFile():
    j = open("Genaioutput.txt", "w")
    j.write(f"\n Output: {convo.last.text}")
    j.close()
    os.startfile("Genaioutput.txt")


def openFileAndGeneratePredict():
    global filepathimg
    filepathimg = filedialog.askopenfilename(initialdir="../Breed_Data",
                                          title="Open file okay?",
                                          filetypes= (("image files","*.jfif"),
                                          ("all files","*.*")))
    readable_prediction(filepathimg)
    output2 = output
    tkinterReadableoutput.set(f"Predicted Breed is {output}")
    convo.send_message(f" imagine you are a expert on dog breeds, give me an informative analysis of the dog breed: {output2}")
    openTextFile()




window = Tk()
window.geometry("1000x600")  # set geometry of window
window.title("Tensorflow Dog Breed Analysis")
window.configure(bg="black")  # change background of window to black
frame = Frame(window)  # inits frame
frame.configure(bg="black") #configure background to black
frame.pack()  # packs frame
tkinterReadableoutput = StringVar()
tkinterReadableoutput.set("No Predicted Breed Yet!")
Label = Label(frame,textvariable=tkinterReadableoutput, fg="black", bg="white", font=("Calibre", 24))  # init label
Label.pack()
button = Button(text="Open",command=openFileAndGeneratePredict)
button.pack()
window.mainloop()





