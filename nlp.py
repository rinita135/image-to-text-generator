# Install required libraries
!pip install pytesseract
!apt-get install tesseract-ocr
!pip install transformers
!pip install torch torchvision
!pip install Pillow
import pytesseract
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from google.colab import files
import matplotlib.pyplot as plt
# Load the pre-trained BLIP model and processor for image captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
def caption_image(image_path):
    # Open the image using PIL
    img = Image.open(image_path)

    # Preprocess the image and generate caption
    inputs = processor(images=img, return_tensors="pt")
    out = model.generate(**inputs)
    
    # Decode the generated caption
    caption = processor.decode(out[0], skip_special_tokens=True)

    return caption
# Upload an image to Colab
uploaded = files.upload()  # This will allow you to upload an image from your computer

# Get the image path
image_path = list(uploaded.keys())[0]  # Getting the uploaded image file name

# Display the uploaded image
img = Image.open(image_path)
plt.imshow(img)
plt.axis('off')  # Hide axes for better viewing
plt.show()

# Get caption for the image
caption = caption_image(image_path)
print("Caption for the image: ", caption)

from IPython.display import display, HTML

html_code = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Captioning in Google Colab</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            text-align: center;
            padding: 50px;
        }
        input[type="file"] {
            margin-top: 20px;
        }
        .output {
            margin-top: 20px;
            font-size: 20px;
            font-weight: bold;
            color: #4CAF50;
        }
        .image-container {
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <h1>Upload an Image for Captioning</h1>
    <p>Upload an image to get a description of its content.</p>
    <input type="file" id="imageUpload" accept="image/*">
    <div id="imageContainer" class="image-container"></div>
    <div id="caption" class="output"></div>

    <script>
        // Function to display the uploaded image and request captioning
        document.getElementById('imageUpload').addEventListener('change', function(event) {
            var reader = new FileReader();
            reader.onload = function(e) {
                var imgElement = document.createElement('img');
                imgElement.src = e.target.result;
                imgElement.width = 400;  // Resize image for better display
                document.getElementById('imageContainer').innerHTML = '';
                document.getElementById('imageContainer').appendChild(imgElement);
                
                // Now make an API call to trigger the backend Python code for captioning
                var formData = new FormData();
                formData.append('image', event.target.files[0]);

                fetch('/predict', {
                    method: 'POST',
                    body: formData,
                })
                .then(response => response.json())
                .then(data => {
                    document.getElementById('caption').textContent = 'Caption: ' + data.caption;
                })
                .catch(error => {
                    console.error('Error:', error);
                });
            };
            reader.readAsDataURL(event.target.files[0]);
        });
    </script>
</body>
</html>
"""

# Display HTML in Colab
display(HTML(html_code))
