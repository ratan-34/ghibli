import os
import uuid
import base64
import cv2
import numpy as np
import tempfile
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from datetime import datetime
from PIL import Image, ImageEnhance
from flask import Flask, request, jsonify, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from flask_cors import CORS
from io import BytesIO

# Initialize Flask app
app = Flask(__name__, template_folder='templates')
CORS(app)

# Config
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'webp'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Bank account details
app.config['BANK_DETAILS'] = {
    'account_name': 'Your Name',
    'account_number': '1234567890',
    'bank_name': 'Your Bank Name',
    'ifsc_code': 'ABCD0123456',
    'upi_id': 'yourname@upi',
    'qr_code_path': os.path.join('static', 'qr_code.png')
}

IMAGE_FOLDER = os.path.join(os.path.dirname(__file__), 'image')
permanent_image = "profile2.jpg"
additional_image = "profile1.jpg"

MODEL_FOLDER = os.path.join(os.path.dirname(__file__), 'models')
GHIBLI_MODEL_PATH = os.path.join(MODEL_FOLDER, 'cartoon_gan_ghibli.pth')

# Improved model architecture with more expressive capacity
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return out + residual  # Skip connection

class EnhancedGhibliGenerator(nn.Module):
    def __init__(self, n_res_blocks=9):  # Increased number of residual blocks
        super(EnhancedGhibliGenerator, self).__init__()
        
        # Initial convolution
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True)
        )
        
        # Downsampling layers
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True)
        )
        
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks
        res_blocks = [ResidualBlock(256) for _ in range(n_res_blocks)]
        self.res_blocks = nn.Sequential(*res_blocks)
        
        # Upsampling layers
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True)
        )
        
        # Output layer
        self.output = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.initial(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.res_blocks(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.output(x)
        return (x + 1) / 2  # Convert from [-1, 1] to [0, 1]

# Enhanced traditional cartoon effect with better edge preservation and color quantization
def enhanced_traditional_cartoon(image):
    # Convert to RGB if needed
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
    # Step 1: Edge detection with better parameters
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                 cv2.THRESH_BINARY, 9, 9)
    
    # Additional edge processing for better line quality
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    edges = 255 - edges  # Invert edges
    
    # Step 2: Color quantization with more bins for better color preservation
    img_small = cv2.resize(image, None, fx=0.5, fy=0.5)
    data = np.float32(img_small.reshape((-1, 3)))
    
    # Increase number of colors for more detail
    K = 12  # Increased from 8
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    _, labels, centers = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    quantized = res.reshape(img_small.shape)
    quantized = cv2.resize(quantized, (image.shape[1], image.shape[0]))
    
    # Step 3: Apply bilateral filter for better smoothing while preserving edges
    color = cv2.bilateralFilter(quantized, 9, 250, 250)
    
    # Step 4: Combine with edges
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    
    # Step 5: Enhance colors
    cartoon_pil = Image.fromarray(cartoon)
    cartoon_pil = ImageEnhance.Color(cartoon_pil).enhance(1.4)
    cartoon_pil = ImageEnhance.Contrast(cartoon_pil).enhance(1.2)
    cartoon_pil = ImageEnhance.Sharpness(cartoon_pil).enhance(1.5)
    
    return np.array(cartoon_pil)

# Global model variable
ghibli_model = None

# Model loading with error handling and fallback
def load_ghibli_model():
    global ghibli_model
    if ghibli_model is None:
        try:
            print(f"Loading model from {GHIBLI_MODEL_PATH}")
            if not os.path.exists(GHIBLI_MODEL_PATH):
                print("Model file not found. Creating a new model.")
                model = EnhancedGhibliGenerator()
                if torch.cuda.is_available():
                    model = model.cuda()
                ghibli_model = model
                return model
            
            # Load pretrained model
            model = EnhancedGhibliGenerator()
            state_dict = torch.load(GHIBLI_MODEL_PATH, map_location=torch.device('cpu'))
            
            # Handle different state_dict formats
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
                
            # Try to load with different key formats (handle module. prefix)
            try:
                model.load_state_dict(state_dict)
            except:
                # Try removing 'module.' prefix if it exists
                new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                model.load_state_dict(new_state_dict, strict=False)
                
            model.eval()
            if torch.cuda.is_available():
                model = model.cuda()
            ghibli_model = model
            print("Ghibli model loaded successfully!")
        except Exception as e:
            print(f"Error loading Ghibli model: {e}")
            ghibli_model = None
            
    return ghibli_model

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def apply_cartoon_effect_traditional(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read image")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply enhanced traditional cartoon effect
        cartoon = enhanced_traditional_cartoon(img)
        
        # Convert back to OpenCV format and encode
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(cartoon, cv2.COLOR_RGB2BGR))
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f"Error in traditional cartoon effect: {e}")
        return None

def apply_ghibli_effect(image_path):
    try:
        # Load image and prepare for model
        image = Image.open(image_path).convert('RGB')
        
        # Improved preprocessing for better quality
        preprocess = transforms.Compose([
            transforms.Resize(512),  # Increase resolution
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        input_tensor = preprocess(image).unsqueeze(0)
        model = load_ghibli_model()
        
        if model is None:
            print("Model not available, falling back to traditional method")
            return None
            
        # Process with model
        with torch.no_grad():
            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()
            output_tensor = model(input_tensor)
            if torch.cuda.is_available():
                output_tensor = output_tensor.cpu()
                
        # Convert output tensor to image
        output_array = output_tensor.squeeze(0).numpy().transpose(1, 2, 0)
        output_array = np.clip(output_array * 255, 0, 255).astype(np.uint8)
        
        # Apply post-processing enhancement
        output_image = Image.fromarray(output_array)
        output_image = ImageEnhance.Color(output_image).enhance(1.3)
        output_image = ImageEnhance.Contrast(output_image).enhance(1.1)
        output_image = ImageEnhance.Sharpness(output_image).enhance(1.2)
        
        # Save to buffer and encode
        buffer = BytesIO()
        output_image.save(buffer, format="JPEG", quality=95)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error in Ghibli effect: {e}")
        return None

def apply_cartoon_effect(image_path, method='ghibli'):
    """
    Apply cartoon effect to an image
    
    Args:
        image_path: Path to input image
        method: 'ghibli' for Ghibli style, 'traditional' for OpenCV-based cartoonization
        
    Returns:
        Base64 encoded string of the processed image
    """
    # Validate the input image
    if not os.path.exists(image_path):
        print(f"Image path does not exist: {image_path}")
        return None
        
    try:
        # Check image validity
        img_test = Image.open(image_path)
        img_test.verify()
    except Exception as e:
        print(f"Invalid image file: {e}")
        return None
    
    # Process based on selected method
    if method == 'ghibli':
        result = apply_ghibli_effect(image_path)
        # Fall back to traditional if ghibli fails
        if result is None:
            print("Ghibli method failed, falling back to traditional method")
            result = apply_cartoon_effect_traditional(image_path)
        return result
    else:
        return apply_cartoon_effect_traditional(image_path)

def image_to_base64(image_path):
    try:
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        print(f"Error reading image: {e}")
        return None

# Add route for processing images
@app.route('/api/cartoon', methods=['POST'])
def cartoon_api():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
        
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
        
    if file and allowed_file(file.filename):
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, filename)
        file.save(temp_path)
        
        # Get method parameter
        method = request.form.get('method', 'ghibli')
        
        # Process image
        result = apply_cartoon_effect(temp_path, method)
        
        # Clean up
        os.remove(temp_path)
        os.rmdir(temp_dir)
        
        if result:
            return jsonify({'image': result})
        else:
            return jsonify({'error': 'Failed to process image'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

# Add home route
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    # Check if running as script with arguments
    import sys
    if len(sys.argv) > 1:
        import argparse
        parser = argparse.ArgumentParser(description='Process images with Ghibli style')
        parser.add_argument('--input', type=str, required=True, help='Input image path')
        parser.add_argument('--output', type=str, required=True, help='Output image path')
        parser.add_argument('--method', type=str, default='ghibli', choices=['ghibli', 'traditional'], 
                            help='Cartoonization method')
        args = parser.parse_args()

        print(f"Processing {args.input} with {args.method} method...")
        result_b64 = apply_cartoon_effect(args.input, method=args.method)
        
        if result_b64:
            with open(args.output, 'wb') as f:
                f.write(base64.b64decode(result_b64))
            print(f"Image processed and saved to {args.output}")
        else:
            print("Failed to process image")
    else:
        # Create models directory if it doesn't exist
        os.makedirs(MODEL_FOLDER, exist_ok=True)
        
        # Start Flask app
        app.run(debug=True, host='0.0.0.0', port=5000)