import streamlit as st
import tensorflow as tf
import os
import numpy as np
from pathlib import Path

# temp_path = Path(__file__).parent / "tempDir"

class_names = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
 
## Page Title
st.set_page_config(page_title = "Image Classification")
st.title("Image Classification")
st.markdown("---")
 
## Sidebar
st.sidebar.header("TF Lite Models")
display = ("Select a Model", "Created FP-16 Quantized Model", "Created Quantized Model", "Created Dynamic Range Quantized Model")
options = list(range(len(display)))
value = st.sidebar.selectbox("Model", options, format_func=lambda x: display[x])
print(value)
 
if value == 1:
    tflite_interpreter = tf.lite.Interpreter(model_path='Models/image_classify.tflite')
    tflite_interpreter.allocate_tensors()
if value == 2:
    tflite_interpreter = tf.lite.Interpreter(model_path='Models/model_int8.tflite')
    tflite_interpreter.allocate_tensors()
if value == 3:
    tflite_interpreter = tf.lite.Interpreter(model_path='Models/model_dynamic.tflite')
    tflite_interpreter.allocate_tensors()
# if value == 4:
#     tflite_interpreter = tf.lite.Interpreter(model_path='Models\created_model_fp16.tflite')
#     tflite_interpreter.allocate_tensors()
# if value == 5:
#     tflite_interpreter = tf.lite.Interpreter(model_path='Models\created_model_int8.tflite')
#     tflite_interpreter.allocate_tensors()
# if value == 6:
#     tflite_interpreter = tf.lite.Interpreter(model_path='Models\created_model_dynamic.tflite')
#     tflite_interpreter.allocate_tensors()
 
def set_input_tensor(interpreter, image):
  """Sets the input tensor."""
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :, :] = image
 
def get_predictions(input_image):
    output_details = tflite_interpreter.get_output_details()
    set_input_tensor(tflite_interpreter, input_image)
    tflite_interpreter.invoke()
    tflite_model_prediction = tflite_interpreter.get_tensor(output_details[0]["index"])
    tflite_model_prediction = tflite_model_prediction.squeeze().argmax(axis = 0)
    print("tflite_model_prediction")
    print(tflite_model_prediction)
    pred_class = class_names[tflite_model_prediction]
    return pred_class
 
 
## Input Fields
uploaded_file = st.file_uploader("Upload a Image", type=["jpg","png", 'jpeg'])
if uploaded_file is not None:
    with open(os.path.join("Models",uploaded_file.name),"wb") as f:
        f.write(uploaded_file.getbuffer())
    path = os.path.join("Models",uploaded_file.name)
    img = tf.keras.preprocessing.image.load_img(path , grayscale=False, color_mode='rgb', target_size=(224,224,3), interpolation='nearest')
    st.image(img)
    print(value)
    # if value == 2 or value == 5:
    #     img = tf.image.convert_image_dtype(img, tf.uint8)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
 
 
if st.button("Get Predictions"):
    suggestion = get_predictions(input_image =img_array)
    st.success(suggestion)