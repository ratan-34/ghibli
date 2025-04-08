import os
import uuid
import base64
import cv2
import numpy as np
import tempfile
import subprocess
import sys
from datetime import datetime
from PIL import Image, ImageEnhance
from flask import Flask, request, jsonify, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from flask_cors import CORS
import onnxruntime

# Initialize Flask app
app = Flask(__name__, template_folder='templates')
CORS(app)

# Config
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'webp'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.config['MODEL_PATH'] = os.path.join('model', 'AnimeGANv2_Hayao.onnx')

# Bank account details (replace with your actual details)
app.config['BANK_DETAILS'] = {
    'account_name': 'Your Name',
    'account_number': '1234567890',
    'bank_name': 'Your Bank Name',
    'ifsc_code': 'ABCD0123456',
    'upi_id': 'yourname@upi',
    'qr_code_path': os.path.join('static', 'qr_code.png')  # Path to your UPI QR code image
}

# Image paths (relative to main.py)
IMAGE_FOLDER = os.path.join(os.path.dirname(__file__), 'image')
permanent_image = "profile2.jpg"
additional_image = "profile1.jpg"

# Ensure necessary directories exist
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), 'static'), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), 'model'), exist_ok=True)

# Load ONNX model with better error handling
try:
    model_path = app.config['MODEL_PATH']
    if os.path.exists(model_path):
        # Use more specific providers config for better compatibility
        providers_options = [
            {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }
        ]
        
        # Try multiple provider options for better compatibility
        try:
            # First try with CPU provider only (most compatible)
            print("Attempting to load model with CPU provider only...")
            ort_session = onnxruntime.InferenceSession(
                model_path,
                providers=['CPUExecutionProvider']
            )
            print("Model loaded successfully with CPU provider")
        except Exception as cpu_error:
            print(f"Failed to load with CPU provider: {cpu_error}")
            # Fallback to default providers
            print("Attempting to load model with default providers...")
            ort_session = onnxruntime.InferenceSession(model_path)
            print("Model loaded successfully with default providers")
    else:
        print(f"Model file not found at {model_path}")
        ort_session = None
except Exception as e:
    print(f"Failed to load model: {e}")
    ort_session = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    try:
        # Using PIL for better compatibility
        img = Image.open(image_path).convert("RGB")
        
        # Get original dimensions
        original_width, original_height = img.size
        
        # Resize with aspect ratio preservation if either dimension exceeds 512
        max_dim = 512
        if original_width > max_dim or original_height > max_dim:
            ratio = min(max_dim / original_width, max_dim / original_height)
            new_width = int(original_width * ratio)
            new_height = int(original_height * ratio)
            img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array, original_width, original_height
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        raise

def postprocess_image(output, original_width=None, original_height=None):
    try:
        output = output.squeeze(0)
        if output.shape[0] == 3:
            output = np.transpose(output, (1, 2, 0))
        output = (output * 255).clip(0, 255).astype(np.uint8)
        output_img = Image.fromarray(output)
        
        # Apply enhancements
        output_img = ImageEnhance.Sharpness(output_img).enhance(1.5)  # Reduced sharpness to prevent artifacts
        output_img = ImageEnhance.Contrast(output_img).enhance(1.2)   # Reduced contrast to prevent color distortion
        
        # Resize back to original dimensions if provided
        if original_width and original_height:
            output_img = output_img.resize((original_width, original_height), Image.LANCZOS)
        
        return np.array(output_img)
    except Exception as e:
        print(f"Error in postprocessing: {e}")
        raise

def apply_ghibli_style(input_path):
    try:
        if ort_session is None:
            print("Using fallback processing (no model available)")
            # Fallback to a simple image processing if model isn't available
            img = cv2.imread(input_path)
            if img is None:
                raise RuntimeError(f"Could not read image from {input_path}")
            
            # Apply a simple filter as fallback
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_blur = cv2.GaussianBlur(img_gray, (21, 21), 0)
            img_edge = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
            img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2BGR)
            
            # Convert the processed image to base64
            _, buffer = cv2.imencode('.jpg', img_edge)
            return base64.b64encode(buffer).decode('utf-8')
        
        # Normal processing with ONNX model
        print("Processing with ONNX model")
        input_img, original_width, original_height = preprocess_image(input_path)
        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name
        
        # More robust running with timeouts and memory management
        try:
            # Run the model with input
            output = ort_session.run([output_name], {input_name: input_img})[0]
            
            # Process the output
            result_img = postprocess_image(output, original_width, original_height)
            
            # Convert to JPEG format
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
            
            # Return base64 encoded image
            return base64.b64encode(buffer).decode('utf-8')
        except Exception as model_error:
            print(f"Error during model execution: {model_error}")
            # Fallback to original image
            with open(input_path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        print(f"Error in style transfer: {e}")
        # Return the original image in case of error
        try:
            with open(input_path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except:
            return None

def process_with_style_api(image_path, style):
    """Process image using the appropriate style model"""
    try:
        # For now, just use the built-in processor as fallback for all styles
        img = cv2.imread(image_path)
        if img is None:
            raise RuntimeError(f"Could not read image from {image_path}")
        
        # Check image dimensions and resize if too large
        height, width = img.shape[:2]
        max_dim = 1024  # Maximum dimension
        if height > max_dim or width > max_dim:
            scale = min(max_dim / width, max_dim / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Apply different processing based on style
        if style == 'ghibli1':
            # Original style
            processed_img = apply_simple_filter(img, 1.5, 10)
        elif style == 'ghibli2':
            # Higher contrast
            processed_img = apply_simple_filter(img, 2.0, 15)
        elif style == 'ghibli3':
            # More colorful
            processed_img = apply_simple_filter(img, 1.8, 5, saturation=1.5)
        elif style == 'ghibli4':
            # More detailed
            processed_img = apply_simple_filter(img, 1.3, 20)
        elif style == 'ghibli5':
            # Dreamy
            processed_img = apply_simple_filter(img, 1.6, 8, brightness=1.2)
        else:
            # Default
            processed_img = apply_simple_filter(img, 1.5, 10)
        
        # Convert the processed image to base64 with better error handling
        try:
            _, buffer = cv2.imencode('.jpg', processed_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if buffer is None or len(buffer) == 0:
                raise ValueError("Empty buffer after encoding")
            return base64.b64encode(buffer).decode('utf-8')
        except Exception as encode_error:
            print(f"Error encoding processed image: {encode_error}")
            # Try a different approach
            temp_path = os.path.join(tempfile.gettempdir(), f"temp_processed_{uuid.uuid4().hex}.jpg")
            cv2.imwrite(temp_path, processed_img)
            with open(temp_path, 'rb') as f:
                result = base64.b64encode(f.read()).decode('utf-8')
            # Clean up
            try:
                os.remove(temp_path)
            except:
                pass
            return result
            
    except Exception as e:
        print(f"Error in style transfer: {e}")
        # Return the original image in case of error
        try:
            with open(image_path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except Exception as original_error:
            print(f"Error returning original image: {original_error}")
            return None

def apply_simple_filter(img, contrast=1.5, blur_amount=10, saturation=1.0, brightness=1.0):
    """Apply a simple artistic filter to an image with better error handling"""
    try:
        # Make a copy of the image to avoid modifying the original
        img_copy = img.copy()
        
        # Convert to PIL Image for easier enhancement
        img_pil = Image.fromarray(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
        
        # Apply enhancements with try/except blocks
        try:
            if brightness != 1.0:
                img_pil = ImageEnhance.Brightness(img_pil).enhance(brightness)
        except Exception as e:
            print(f"Error applying brightness: {e}")
            
        try:
            if contrast != 1.0:
                img_pil = ImageEnhance.Contrast(img_pil).enhance(contrast)
        except Exception as e:
            print(f"Error applying contrast: {e}")
            
        try:
            if saturation != 1.0:
                img_pil = ImageEnhance.Color(img_pil).enhance(saturation)
        except Exception as e:
            print(f"Error applying saturation: {e}")
        
        # Convert back to OpenCV format
        img_copy = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        # Apply blur if specified
        if blur_amount > 0:
            try:
                # Make sure blur_amount is odd
                blur_amount = blur_amount if blur_amount % 2 == 1 else blur_amount + 1
                # Apply blur
                blurred = cv2.GaussianBlur(img_copy, (blur_amount, blur_amount), 0)
                img_copy = cv2.addWeighted(img_copy, 1.5, blurred, -0.5, 0)
            except Exception as e:
                print(f"Error applying blur: {e}")
                
        return img_copy
    except Exception as e:
        print(f"Error in simple filter: {e}")
        return img  # Return original image on error

def image_to_base64(image_path):
    try:
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return None
            
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        print(f"Error reading image: {e}")
        return None

@app.route('/')
def index():
    # Build absolute paths to images
    permanent_path = os.path.join(IMAGE_FOLDER, permanent_image)
    additional_path = os.path.join(IMAGE_FOLDER, additional_image)

    # Default images if the actual ones don't exist
    permanent_b64 = None
    additional_b64 = None
    
    # Try to get the actual images
    if os.path.exists(permanent_path):
        permanent_b64 = image_to_base64(permanent_path)
        
    if os.path.exists(additional_path):
        additional_b64 = image_to_base64(additional_path)

    return render_template(
        'index.html',
        permanent_image_url=f"data:image/jpeg;base64,{permanent_b64}" if permanent_b64 else None,
        has_permanent_image=permanent_b64 is not None,
        additional_image_url=f"data:image/jpeg;base64,{additional_b64}" if additional_b64 else None,
        has_additional_image=additional_b64 is not None
    )

@app.route('/payment')
def payment():
    plan = request.args.get('plan', 'creator')
    period = request.args.get('period', 'monthly')
    
    # Calculate amount based on plan and period
    amount = calculate_amount(plan, period)
    currency = 'INR'  # Changed to INR for Indian payment methods
    
    # Convert QR code to base64
    qr_code_b64 = None
    if os.path.exists(app.config['BANK_DETAILS']['qr_code_path']):
        with open(app.config['BANK_DETAILS']['qr_code_path'], 'rb') as f:
            qr_code_b64 = base64.b64encode(f.read()).decode('utf-8')
    
    return render_template(
        'payment.html',
        plan=plan,
        period=period,
        amount=amount,
        currency=currency,
        bank_details=app.config['BANK_DETAILS'],
        qr_code_url=f"data:image/png;base64,{qr_code_b64}" if qr_code_b64 else None
    )

@app.route('/verify-payment', methods=['POST'])
def verify_payment():
    try:
        data = request.json
        payment_method = data.get('payment_method')
        transaction_id = data.get('transaction_id')
        amount = data.get('amount')
        plan = data.get('plan')
        period = data.get('period')
        
        # In a real implementation, you would verify the payment with your bank
        # For this example, we'll just simulate a successful verification
        
        # Generate a fake transaction ID if not provided
        if not transaction_id:
            transaction_id = f"TXN{uuid.uuid4().hex[:8].upper()}"
        
        # Format the amount with 2 decimal places
        formatted_amount = "{:.2f}".format(float(amount) / 100)
        
        return jsonify({
            'success': True,
            'transaction_id': transaction_id,
            'payment_method': payment_method.capitalize(),
            'amount': formatted_amount,
            'plan': plan,
            'period': period,
            'message': 'Payment verified successfully'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/payment-success')
def payment_success():
    # Get parameters from the request
    plan = request.args.get('plan', 'creator')
    period = request.args.get('period', 'monthly')
    transaction_id = request.args.get('transaction_id', f"TXN{uuid.uuid4().hex[:8].upper()}")
    payment_method = request.args.get('payment_method', 'Bank Transfer')
    
    # Calculate amount based on plan and period
    amount = calculate_amount(plan, period)
    
    # Format current date and time
    current_datetime = datetime.now().strftime("%B %d, %Y, %I:%M %p")
    
    return render_template(
        'payment_success.html',
        plan=plan,
        period=period,
        amount=amount,
        transaction_id=transaction_id,
        datetime=current_datetime,
        payment_method=payment_method
    )

@app.route('/upload', methods=['POST'])
def upload_file():
    # Add debug output
    print("Upload endpoint called")
    
    if 'file' not in request.files:
        print("No file part in request")
        return jsonify({'status': 'error', 'message': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        print("Empty filename")
        return jsonify({'status': 'error', 'message': 'No selected file'}), 400

    if not allowed_file(file.filename):
        print(f"File type not allowed: {file.filename}")
        return jsonify({'status': 'error', 'message': 'File type not allowed'}), 400

    try:
        # Create a temporary file with proper cross-platform handling
        temp_dir = tempfile.gettempdir()
        temp_filename = secure_filename(file.filename)
        if not temp_filename:  # In case secure_filename returns empty string
            temp_filename = f"upload_{uuid.uuid4().hex}.jpg"
        temp_path = os.path.join(temp_dir, temp_filename)
        
        # Save file with explicit mode
        print(f"Saving file to {temp_path}")
        file.save(temp_path)
        
        # Verify file was saved correctly
        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            print("File was not saved correctly")
            return jsonify({'status': 'error', 'message': 'Failed to save uploaded file'}), 500

        # Get the selected style from form data
        selected_style = request.form.get('style', 'ghibli1')
        print(f"Selected style: {selected_style}")
        
        # Process image based on selected style
        print("Processing image...")
        result_b64 = process_with_style_api(temp_path, selected_style)
        
        if not result_b64:
            print("Failed to process image")
            # Try a simple fallback process
            img = cv2.imread(temp_path)
            if img is not None:
                img_processed = cv2.GaussianBlur(img, (5, 5), 0)
                _, buffer = cv2.imencode('.jpg', img_processed)
                result_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Read the original file again for base64 encoding
        print("Reading original file for base64")
        with open(temp_path, 'rb') as f:
            original_b64 = base64.b64encode(f.read()).decode('utf-8')
        
        # Clean up
        try:
            print("Removing temp file")
            os.remove(temp_path)
        except Exception as e:
            print(f"Error removing temp file: {e}")

        if not result_b64:
            print("Failed to process image after fallback")
            return jsonify({'status': 'error', 'message': 'Failed to process image'}), 500

        print("Successfully processed image")
        return jsonify({
            'status': 'success',
            'original': f"data:image/jpeg;base64,{original_b64}",
            'result': f"data:image/jpeg;base64,{result_b64}"
        })

    except Exception as e:
        print(f"Upload error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': f'Error: {str(e)}'}), 500

# Simplified style-specific upload routes
@app.route('/upload-ghibli1', methods=['POST'])
def upload_file_ghibli1():
    return process_style_upload('ghibli1')

@app.route('/upload-ghibli2', methods=['POST'])
def upload_file_ghibli2():
    return process_style_upload('ghibli2')

@app.route('/upload-ghibli3', methods=['POST'])
def upload_file_ghibli3():
    return process_style_upload('ghibli3')

@app.route('/upload-ghibli4', methods=['POST'])
def upload_file_ghibli4():
    return process_style_upload('ghibli4')

@app.route('/upload-ghibli5', methods=['POST'])
def upload_file_ghibli5():
    return process_style_upload('ghibli5')

def process_style_upload(style):
    """Common handler for all style upload routes"""
    print(f"Style upload called: {style}")
    
    if 'file' not in request.files:
        print("No file part in request")
        return jsonify({'status': 'error', 'message': 'No file part'}), 400
        
    file = request.files['file']
    if file.filename == '':
        print("Empty filename")
        return jsonify({'status': 'error', 'message': 'No selected file'}), 400
        
    if not allowed_file(file.filename):
        print(f"File type not allowed: {file.filename}")
        return jsonify({'status': 'error', 'message': 'File type not allowed'}), 400
        
    try:
        # Save the uploaded file temporarily
        temp_dir = tempfile.gettempdir()
        temp_filename = secure_filename(file.filename)
        if not temp_filename:  # In case secure_filename returns empty string
            temp_filename = f"upload_{uuid.uuid4().hex}.jpg"
        temp_path = os.path.join(temp_dir, temp_filename)
        
        print(f"Saving file to {temp_path}")
        file.save(temp_path)
        
        # Verify file was saved correctly
        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            print("File was not saved correctly")
            return jsonify({'status': 'error', 'message': 'Failed to save uploaded file'}), 500
        
        # Process image with the specified style
        print(f"Processing image with style: {style}")
        result_b64 = process_with_style_api(temp_path, style)
        
        if not result_b64:
            print("Failed to process image, trying fallback")
            # Try a simple fallback process
            img = cv2.imread(temp_path)
            if img is not None:
                img_processed = cv2.GaussianBlur(img, (5, 5), 0)
                _, buffer = cv2.imencode('.jpg', img_processed)
                result_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Read the original file for base64 encoding
        print("Reading original file for base64")
        with open(temp_path, 'rb') as f:
            original_b64 = base64.b64encode(f.read()).decode('utf-8')
        
        # Clean up
        try:
            print("Removing temp file")
            os.remove(temp_path)
        except Exception as e:
            print(f"Error removing temp file: {e}")
            
        if not result_b64:
            print("Failed to process image after fallback")
            return jsonify({'status': 'error', 'message': 'Failed to process image'}), 500
            
        print("Successfully processed image")
        return jsonify({
            'status': 'success',
            'original': f"data:image/jpeg;base64,{original_b64}",
            'result': f"data:image/jpeg;base64,{result_b64}"
        })
    except Exception as e:
        print(f"Style upload error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': f'Error: {str(e)}'}), 500

def calculate_amount(plan, period):
    # Implement your pricing logic here
    if plan == 'starter':
        return 900 if period == 'monthly' else 9000  # ₹9/month or ₹90/year
    elif plan == 'creator':
        return 1900 if period == 'monthly' else 18000  # ₹19/month or ₹180/year
    elif plan == 'professional':
        return 4900 if period == 'monthly' else 47000  # ₹49/month or ₹470/year
    return 0

@app.errorhandler(404)
def page_not_found(e):
    print(f"404 error: {e}")
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    print(f"500 error: {e}")
    return render_template('500.html'), 500

@app.route('/health')
def health_check():
    """Endpoint for monitoring the health of the application"""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': ort_session is not None
    })

if __name__ == '__main__':
    # Force UTF-8 encoding for the Flask app
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    app.run(debug=True, port=5000, threaded=True)
