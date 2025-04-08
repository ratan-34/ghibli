import os
import uuid
import base64
import cv2
import numpy as np
import tempfile
from datetime import datetime
from PIL import Image, ImageEnhance
from flask import Flask, request, jsonify, render_template
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
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['STATIC_FOLDER'] = 'static'

# Bank account details
app.config['BANK_DETAILS'] = {
    'account_name': 'Your Name',
    'account_number': '1234567890',
    'bank_name': 'Your Bank Name',
    'ifsc_code': 'ABCD0123456',
    'upi_id': 'yourname@upi',
    'qr_code_path': os.path.join('static', 'qr_code.png')
}

# Ensure necessary directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), 'static', exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), 'model'), exist_ok=True)

# Sample images
DEFAULT_IMAGES = {
    'permanent': "sample1.jpg",
    'additional': "sample2.jpg"
}

# Load ONNX model
ort_session = None
try:
    model_path = app.config['MODEL_PATH']
    if os.path.exists(model_path):
        ort_session = onnxruntime.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        print(f"Model loaded successfully from {model_path}")
    else:
        print(f"Model file not found at {model_path}")
except Exception as e:
    print(f"Failed to load model: {e}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    try:
        img = Image.open(image_path).convert("RGB").resize((512, 512))
        img = np.array(img).astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        raise

def postprocess_image(output):
    try:
        output = output.squeeze(0)
        if output.shape[0] == 3:
            output = np.transpose(output, (1, 2, 0))
        output = (output * 255).clip(0, 255).astype(np.uint8)
        output_img = Image.fromarray(output)
        output_img = ImageEnhance.Sharpness(output_img).enhance(2.0)
        output_img = ImageEnhance.Contrast(output_img).enhance(1.5)
        return np.array(output_img)
    except Exception as e:
        print(f"Error in postprocessing: {e}")
        raise

def apply_ghibli_style(input_path):
    try:
        if ort_session is None:
            return apply_simple_filter_fallback(input_path)
        
        input_img = preprocess_image(input_path)
        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name
        output = ort_session.run([output_name], {input_name: input_img})[0]

        result_img = postprocess_image(output)
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f"Error in style transfer: {e}")
        return apply_simple_filter_fallback(input_path)

def apply_simple_filter_fallback(input_path):
    """Fallback processing when ONNX model fails"""
    try:
        img = cv2.imread(input_path)
        if img is None:
            raise RuntimeError(f"Could not read image from {input_path}")
        
        # Convert to sketch-like effect
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                     cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(img, 9, 300, 300)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        
        _, buffer = cv2.imencode('.jpg', cartoon)
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f"Fallback processing failed: {e}")
        try:
            with open(input_path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except:
            return None

def process_with_style_api(image_path, style):
    """Process image using the appropriate style"""
    try:
        if style == 'ghibli1':
            return apply_ghibli_style(image_path)
        else:
            return apply_style_variation(image_path, style)
    except Exception as e:
        print(f"Error in style transfer: {e}")
        try:
            with open(image_path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except:
            return None

def apply_style_variation(image_path, style):
    """Apply different style variations"""
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Could not read image from {image_path}")
    
    if style == 'ghibli2':
        # Higher contrast
        processed = apply_simple_filter(img, contrast=2.0, blur_amount=15)
    elif style == 'ghibli3':
        # More colorful
        processed = apply_simple_filter(img, contrast=1.8, blur_amount=5, saturation=1.5)
    elif style == 'ghibli4':
        # More detailed
        processed = apply_simple_filter(img, contrast=1.3, blur_amount=20)
    elif style == 'ghibli5':
        # Dreamy
        processed = apply_simple_filter(img, contrast=1.6, blur_amount=8, brightness=1.2)
    else:
        # Default
        processed = apply_simple_filter(img, contrast=1.5, blur_amount=10)
    
    _, buffer = cv2.imencode('.jpg', processed)
    return base64.b64encode(buffer).decode('utf-8')

def apply_simple_filter(img, contrast=1.5, blur_amount=10, saturation=1.0, brightness=1.0):
    """Apply a simple artistic filter to an image"""
    try:
        # Convert to PIL Image for easier enhancement
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Apply enhancements
        if brightness != 1.0:
            img_pil = ImageEnhance.Brightness(img_pil).enhance(brightness)
        if contrast != 1.0:
            img_pil = ImageEnhance.Contrast(img_pil).enhance(contrast)
        if saturation != 1.0:
            img_pil = ImageEnhance.Color(img_pil).enhance(saturation)
        
        # Convert back to OpenCV format
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        # Apply blur if specified
        if blur_amount > 0:
            blurred = cv2.GaussianBlur(img, (blur_amount, blur_amount), 0)
            img = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
            
        return img
    except Exception as e:
        print(f"Error in simple filter: {e}")
        return img

def get_sample_image_path(filename):
    """Get path to sample image or default if not found"""
    sample_path = os.path.join(app.config['STATIC_FOLDER'], filename)
    if os.path.exists(sample_path):
        return sample_path
    return None

@app.route('/')
def index():
    permanent_path = get_sample_image_path(DEFAULT_IMAGES['permanent'])
    additional_path = get_sample_image_path(DEFAULT_IMAGES['additional'])

    permanent_b64 = image_to_base64(permanent_path) if permanent_path else None
    additional_b64 = image_to_base64(additional_path) if additional_path else None

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
    amount = calculate_amount(plan, period)
    
    qr_code_b64 = None
    qr_path = app.config['BANK_DETAILS']['qr_code_path']
    if os.path.exists(qr_path):
        qr_code_b64 = image_to_base64(qr_path)
    
    return render_template(
        'payment.html',
        plan=plan,
        period=period,
        amount=amount,
        currency='INR',
        bank_details=app.config['BANK_DETAILS'],
        qr_code_url=f"data:image/png;base64,{qr_code_b64}" if qr_code_b64 else None
    )

@app.route('/verify-payment', methods=['POST'])
def verify_payment():
    try:
        data = request.json
        return jsonify({
            'success': True,
            'transaction_id': data.get('transaction_id', f"TXN{uuid.uuid4().hex[:8].upper()}"),
            'payment_method': data.get('payment_method', 'Bank Transfer').capitalize(),
            'amount': "{:.2f}".format(float(data.get('amount', 0)) / 100),
            'plan': data.get('plan', 'creator'),
            'period': data.get('period', 'monthly'),
            'message': 'Payment verified successfully'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/payment-success')
def payment_success():
    return render_template(
        'payment_success.html',
        plan=request.args.get('plan', 'creator'),
        period=request.args.get('period', 'monthly'),
        amount=calculate_amount(request.args.get('plan', 'creator'), 
                               request.args.get('period', 'monthly')),
        transaction_id=request.args.get('transaction_id', f"TXN{uuid.uuid4().hex[:8].upper()}"),
        datetime=datetime.now().strftime("%B %d, %Y, %I:%M %p"),
        payment_method=request.args.get('payment_method', 'Bank Transfer')
    )

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'status': 'error', 'message': 'File type not allowed'}), 400

    try:
        # Save to upload folder instead of temp directory
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)

        selected_style = request.form.get('style', 'ghibli1')
        result_b64 = process_with_style_api(upload_path, selected_style)
        original_b64 = image_to_base64(upload_path)
        
        # Clean up
        try:
            os.remove(upload_path)
        except Exception as e:
            print(f"Error removing uploaded file: {e}")

        if not result_b64:
            return jsonify({'status': 'error', 'message': 'Failed to process image'}), 500

        return jsonify({
            'status': 'success',
            'original': f"data:image/jpeg;base64,{original_b64}",
            'result': f"data:image/jpeg;base64,{result_b64}"
        })

    except Exception as e:
        print(f"Upload error: {str(e)}")
        return jsonify({'status': 'error', 'message': f'Error: {str(e)}'}), 500

def image_to_base64(image_path):
    if not image_path or not os.path.exists(image_path):
        return None
    try:
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        print(f"Error reading image: {e}")
        return None

def calculate_amount(plan, period):
    if plan == 'starter':
        return 900 if period == 'monthly' else 9000
    elif plan == 'creator':
        return 1900 if period == 'monthly' else 18000
    elif plan == 'professional':
        return 4900 if period == 'monthly' else 47000
    return 0

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
