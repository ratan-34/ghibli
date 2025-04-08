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

# Configuration
app.config.update({
    'ALLOWED_EXTENSIONS': {'png', 'jpg', 'jpeg', 'webp'},
    'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,
    'MODEL_PATH': os.path.join('models', 'ghibli_style.onnx'),
    'UPLOAD_FOLDER': os.path.join('static', 'uploads'),
    'OUTPUT_FOLDER': os.path.join('static', 'outputs'),
    # Bank account details
    'BANK_DETAILS': {
        'account_name': 'Your Name',
        'account_number': '1234567890',
        'bank_name': 'Your Bank Name',
        'ifsc_code': 'ABCD0123456',
        'upi_id': 'yourname@upi',
        'qr_code_path': os.path.join('static', 'qr_code.png')
    }
})

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Image paths (relative to main.py)
IMAGE_FOLDER = os.path.join(os.path.dirname(__file__), 'image')
permanent_image = "profile2.jpg"
additional_image = "profile1.jpg"

# Load ONNX model
def load_model():
    try:
        providers = ['CPUExecutionProvider']
        if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')
        
        session = onnxruntime.InferenceSession(
            app.config['MODEL_PATH'],
            providers=providers
        )
        print("Ghibli style model loaded successfully")
        return session
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

ort_session = load_model()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        # Maintain aspect ratio while resizing
        width, height = img.size
        if width > height:
            new_width = 512
            new_height = int(height * (512 / width))
        else:
            new_height = 512
            new_width = int(width * (512 / height))
        
        img = img.resize((new_width, new_height), Image.LANCZOS)
        img = np.array(img).astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        print(f"Preprocessing error: {str(e)}")
        raise

def postprocess_image(output):
    try:
        output = output.squeeze(0)
        if output.shape[0] == 3:
            output = np.transpose(output, (1, 2, 0))
        output = (output * 255).clip(0, 255).astype(np.uint8)
        
        # Apply Ghibli-style enhancements
        img = Image.fromarray(output)
        
        # Enhance colors and contrast
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.2)
        
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.1)
        
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.5)
        
        return np.array(img)
    except Exception as e:
        print(f"Postprocessing error: {str(e)}")
        raise

def apply_ghibli_style(input_path, output_path=None):
    try:
        if ort_session is None:
            raise RuntimeError("Model not loaded")
        
        input_img = preprocess_image(input_path)
        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name
        
        # Run inference
        output = ort_session.run([output_name], {input_name: input_img})[0]
        
        # Post-process
        result_img = postprocess_image(output)
        
        # Save or return the result
        if output_path:
            cv2.imwrite(output_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
            return output_path
        else:
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
            return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f"Style transfer error: {str(e)}")
        return None

def process_with_style_api(image_path, style):
    """Process image using the appropriate style model"""
    try:
        style_map = {
            'ghibli1': 'main.py',
            'ghibli2': 'main1.py',
            'ghibli3': 'main2.py',
            'ghibli4': 'main3.py',
            'ghibli5': 'main4.py'
        }
        
        script_name = style_map.get(style, 'main.py')
        output_dir = tempfile.gettempdir()
        output_path = os.path.join(output_dir, f"output_{uuid.uuid4().hex}.jpg")
        
        python_exec = sys.executable or 'python'
        command = f"{python_exec} {script_name} --input {image_path} --output {output_path}"
        
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=os.environ.copy()
        )
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            raise RuntimeError(f"Script failed: {stderr.decode('utf-8', errors='replace')}")
        
        if not os.path.exists(output_path):
            raise RuntimeError("Output file not created by script")
            
        with open(output_path, 'rb') as f:
            result_b64 = base64.b64encode(f.read()).decode('utf-8')
        
        try:
            os.remove(output_path)
        except Exception as e:
            print(f"Error removing output file: {e}")
        
        return result_b64
        
    except Exception as e:
        print(f"Error in style transfer: {e}")
        return None

def image_to_base64(image_path):
    try:
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        print(f"Error reading image: {e}")
        return None

@app.route('/')
def index():
    permanent_path = os.path.join(IMAGE_FOLDER, permanent_image)
    additional_path = os.path.join(IMAGE_FOLDER, additional_image)

    permanent_b64 = image_to_base64(permanent_path)
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
    amount = calculate_amount(plan, period)
    currency = 'INR'
    
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
        transaction_id = data.get('transaction_id', f"TXN{uuid.uuid4().hex[:8].upper()}")
        amount = data.get('amount')
        plan = data.get('plan')
        period = data.get('period')
        
        formatted_amount = "{:.2f}".format(amount / 100)
        
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
    plan = request.args.get('plan', 'creator')
    period = request.args.get('period', 'monthly')
    transaction_id = request.args.get('transaction_id', f"TXN{uuid.uuid4().hex[:8].upper()}")
    payment_method = request.args.get('payment_method', 'Bank Transfer')
    
    amount = calculate_amount(plan, period)
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
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'status': 'error', 'message': 'File type not allowed'}), 400

    try:
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, secure_filename(file.filename))
        file.save(temp_path)

        selected_style = request.form.get('style', 'ghibli1')
        
        if selected_style == 'ghibli1':
            result_b64 = apply_ghibli_style(temp_path)
        else:
            result_b64 = process_with_style_api(temp_path, selected_style)
        
        with open(temp_path, 'rb') as f:
            original_b64 = base64.b64encode(f.read()).decode('utf-8')
        
        try:
            os.remove(temp_path)
        except Exception as e:
            print(f"Error removing temp file: {e}")

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

@app.route('/upload-ghibli1', methods=['POST'])
def upload_file_ghibli1():
    return process_with_style('main.py')

@app.route('/upload-ghibli2', methods=['POST'])
def upload_file_ghibli2():
    return process_with_style('main1.py')

@app.route('/upload-ghibli3', methods=['POST'])
def upload_file_ghibli3():
    return process_with_style('main2.py')

@app.route('/upload-ghibli4', methods=['POST'])
def upload_file_ghibli4():
    return process_with_style('main3.py')

@app.route('/upload-ghibli5', methods=['POST'])
def upload_file_ghibli5():
    return process_with_style('main4.py')

def process_with_style(script_name):
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'status': 'error', 'message': 'File type not allowed'}), 400
    try:
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, secure_filename(file.filename))
        file.save(temp_path)
        
        output_path = os.path.join(temp_dir, "output.jpg")
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        command = f"python {script_name} --input {temp_path}"
        
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env
        )
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            raise RuntimeError(f"Script failed: {stderr.decode('utf-8', errors='replace')}")
        
        if not os.path.exists(output_path):
            raise RuntimeError("Output file not created by script")
            
        with open(output_path, 'rb') as f:
            result_b64 = base64.b64encode(f.read()).decode('utf-8')
        
        with open(temp_path, 'rb') as f:
            original_b64 = base64.b64encode(f.read()).decode('utf-8')
        
        try:
            os.remove(temp_path)
            os.remove(output_path)
        except Exception as e:
            print(f"Error removing temp files: {e}")
            
        return jsonify({
            'status': 'success',
            'original': f"data:image/jpeg;base64,{original_b64}",
            'result': f"data:image/jpeg;base64,{result_b64}"
        })
    except Exception as e:
        print(f"Upload error: {str(e)}")
        return jsonify({'status': 'error', 'message': f'Error: {str(e)}'}), 500

def calculate_amount(plan, period):
    if plan == 'starter':
        return 900 if period == 'monthly' else 9000
    elif plan == 'creator':
        return 1900 if period == 'monthly' else 18000
    elif plan == 'professional':
        return 4900 if period == 'monthly' else 47000
    return 0

if __name__ == '__main__':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
