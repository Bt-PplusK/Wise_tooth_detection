from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tflite.model

app = Flask(__name__)

def predict_tooth(image_path):
  """
  This function takes an image path, preprocesses it, and returns the predicted presence/absence of a wise tooth.
  """
  img = Image.open(image_path).convert("L")
  img = np.array(img) / 255.0  # Normalize pixel values
  img = img.reshape((1, *img.shape))  # Add batch dimension

  interpreter = tflite.model.Model.from_file("model.tflite")
  interpreter.allocate_tensors()

  input_details = interpreter.inputs[0]
  output_details = interpreter.outputs[0]
  interpreter.set_tensor(input_details['index'], img)
  interpreter.invoke()
  prediction = interpreter.get_tensor(output_details['index'])[0][0]

  if prediction >= 0.5:
    return "Wise tooth present"
  else:
    return "No wise tooth detected"

@app.route("/predict", methods=["POST"])
def predict_endpoint():
  if "image" not in request.files:
    return jsonify({"error": "No image uploaded"}), 400

  image_file = request.files["image"]
  image_path = f"209.jpg"  # Save uploaded image
  image_file.save(image_path)

  # Use the predict_tooth function to get the prediction result
  prediction = predict_tooth(image_path)

  return jsonify({"prediction": prediction})

if __name__ == "__main__":
  app.run(debug=True)
