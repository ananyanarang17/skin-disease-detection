# -------------------- IMPORTS -------------------- #
from flask import Flask, render_template, request, redirect, url_for, jsonify, session
from flask_login import LoginManager, UserMixin
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from fpdf import FPDF
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import cv2
import os
import datetime
import requests
import re
import csv
import base64
import hashlib  
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

# flask initialize
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['SECRET_KEY'] = 'your-secret-key-change-this-in-production-2024'

#  CONFIGURE DATABASE
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///skin_ai.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

#LOGIN MANAGER 
login_manager = LoginManager()
login_manager.init_app(app)

class DummyUser(UserMixin):
    def __init__(self, id="guest", username="Guest"):
        self.id = id
        self.username = username

    @property
    def is_authenticated(self):
        return False

@login_manager.user_loader
def load_user(user_id):
    return DummyUser()

#  DATABASE MODELS 
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.now)

class Appointment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_name = db.Column(db.String(100))
    user_email = db.Column(db.String(120))
    date = db.Column(db.String(20))
    time = db.Column(db.String(20))
    doctor = db.Column(db.String(100))
    status = db.Column(db.String(50), default="Pending")

class Report(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_name = db.Column(db.String(100))
    disease = db.Column(db.String(100))
    confidence = db.Column(db.Float)
    image_path = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.datetime.now)

class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    message = db.Column(db.Text, nullable=False)
    feedback_type = db.Column(db.String(50), nullable=True)
    rating = db.Column(db.Integer, nullable=True)
    device_type = db.Column(db.String(50), nullable=True)
    page = db.Column(db.String(50), nullable=True)
    consent = db.Column(db.Boolean, default=False)
    image_path = db.Column(db.String(255), nullable=True)
    submitted_at = db.Column(db.DateTime, default=datetime.datetime.now)

# LOAD MODEL 
model = load_model("best_finetuned_model.h5")

label_map = {
    0: 'akiec - Actinic keratoses',
    1: 'bcc - Basal cell carcinoma',
    2: 'bkl - Benign keratosis-like lesions',
    3: 'df - Dermatofibroma',
    4: 'mel - Melanoma',
    5: 'nv - Melanocytic nevi',
    6: 'vasc - Vascular lesions'
}

info = {
    "akiec": {"desc": "Precancerous skin lesion due to sun damage.",
              "tip": "Consult a dermatologist for cryotherapy or minor removal."},
    "bcc": {"desc": "Most common skin cancer, slow growing.",
            "tip": "Usually treated by surgical excision or topical medicine."},
    "bkl": {"desc": "Benign age-related growths or sun spots.",
            "tip": "Generally harmless; use sunscreen daily."},
    "df": {"desc": "Benign fibrous skin bump caused by small injuries.",
           "tip": "Safe; removal is optional."},
    "mel": {"desc": "Serious skin cancer caused by UV exposure.",
            "tip": "Immediate dermatology consultation is recommended."},
    "nv": {"desc": "Common mole formed by clusters of pigment cells.",
           "tip": "Normal, monitor for changes in color or shape."},
    "vasc": {"desc": "Blood vessel growth (angioma).",
             "tip": "Usually harmless; removal is cosmetic."}
}

history = []

# 🔐 VALIDATION FUNCTIONS

def validate_email(email):
    """Check if email format is valid"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_password(password):
    """Password must be at least 6 characters"""
    return len(password) >= 6

def hash_password(password):
    """
    Custom password hashing using SHA256
    Works on all Python versions without dependency issues
    """
    salt = os.urandom(32)
    pwdhash = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt,
        100000  
    )
    return base64.b64encode(salt + pwdhash).decode('ascii')

def verify_password(stored_password, provided_password):
    """
    Verify a stored password against provided password
    """
    try:
        decoded = base64.b64decode(stored_password.encode('ascii'))
        salt = decoded[:32]
        stored_hash = decoded[32:]
        pwdhash = hashlib.pbkdf2_hmac(
            'sha256',
            provided_password.encode('utf-8'),
            salt,
            100000
        )
        return pwdhash == stored_hash
    except Exception as e:
        print(f"Password verification error: {e}")
        return False


def get_current_user():
    """
    Returns current logged-in user or None
    Use this in other routes to check if user is logged in
    """
    if 'user_id' in session:
        return User.query.get(session['user_id'])
    return None

# AUTHENTICATION ROUTES
@app.route('/signup', methods=['POST'])
def signup():
    """
    Handles user registration
    """
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')

        print(f"📝 Signup attempt: {username} ({email})")

        if not username or not email or not password:
            return jsonify({
                'success': False,
                'message': '⚠️ All fields are required!'
            }), 400

        if len(username) < 3:
            return jsonify({
                'success': False,
                'message': '⚠️ Username must be at least 3 characters!'
            }), 400

        if not validate_email(email):
            return jsonify({
                'success': False,
                'message': '⚠️ Invalid email format!'
            }), 400

        if not validate_password(password):
            return jsonify({
                'success': False,
                'message': '⚠️ Password must be at least 6 characters!'
            }), 400

        existing_user = User.query.filter(
            (User.username == username) | (User.email == email)
        ).first()

        if existing_user:
            if existing_user.username == username:
                return jsonify({
                    'success': False,
                    'message': '❌ Username already taken!'
                }), 409
            else:
                return jsonify({
                    'success': False,
                    'message': '❌ Email already registered!'
                }), 409

        password_hash = hash_password(password)
        new_user = User(
            username=username,
            email=email,
            password=password_hash
        )
        db.session.add(new_user)
        db.session.commit()

        session['user_id'] = new_user.id
        session['username'] = new_user.username
        session['email'] = new_user.email

        print(f"✅ New user created: {username}")

        return jsonify({
            'success': True,
            'message': '✅ Account created successfully!',
            'user': {
                'id': new_user.id,
                'username': new_user.username,
                'email': new_user.email
            }
        }), 201

    except Exception as e:
        db.session.rollback()
        print(f"❌ Signup error: {e}")
        return jsonify({
            'success': False,
            'message': '❌ Server error. Please try again.'
        }), 500

@app.route('/login', methods=['POST'])
def login():
    """
    Handles user login
    """
    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')

        print(f"🔑 Login attempt: {email}")

        if not email or not password:
            return jsonify({
                'success': False,
                'message': '⚠️ Email and password are required!'
            }), 400

        user = User.query.filter_by(email=email).first()

        if not user:
            return jsonify({
                'success': False,
                'message': '❌ No account found with this email!'
            }), 401

        if not verify_password(user.password, password):
            return jsonify({
                'success': False,
                'message': '❌ Incorrect password!'
            }), 401

        session['user_id'] = user.id
        session['username'] = user.username
        session['email'] = user.email

        print(f"✅ User logged in: {user.username}")

        return jsonify({
            'success': True,
            'message': f'✅ Welcome back, {user.username}!',
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email
            }
        }), 200

    except Exception as e:
        print(f"❌ Login error: {e}")
        return jsonify({
            'success': False,
            'message': '❌ Server error. Please try again.'
        }), 500

@app.route('/logout', methods=['POST'])
def logout():
    """
    Logs user out by clearing session
    """
    username = session.get('username', 'Unknown')
    session.clear()
    print(f"👋 User logged out: {username}")
    
    return jsonify({
        'success': True,
        'message': '✅ Logged out successfully!'
    }), 200

@app.route('/check-auth', methods=['GET'])
def check_auth():
    """
    Checks if user is currently logged in
    """
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        if user:
            return jsonify({
                'authenticated': True,
                'user': {
                    'id': user.id,
                    'username': user.username,
                    'email': user.email
                }
            }), 200
    
    return jsonify({
        'authenticated': False
    }), 200

# PAGE ROUTES

@app.route('/')
def home():
    return render_template('index.html', current_user=DummyUser())

@app.route("/appointments")
def appointments():
    return render_template("appointments.html")

@app.route("/tips")
def tips():
    return render_template("tips.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/clinics")
def clinics():
    clinics_data = [
        {
            "name": "🏥 Dermacare Jaipur",
            "address": "Sardar Patel Marg, C-Scheme, Jaipur, Rajasthan 302001",
            "contact": "+91 (141) 237-7890",
            "timing": "Mon–Fri: 10 AM – 7 PM, Sat: 10 AM – 6 PM, Sun: Closed"
        },
        {
            "name": "🏥 Skin Clinic Jaipur",
            "address": "MI Road, Near Ajmera Circle, Jaipur, Rajasthan 302005",
            "contact": "+91 (141) 256-8934",
            "timing": "Mon–Sun: 9 AM – 9 PM"
        },
        {
            "name": "🏥 Apollo Dermatology Center",
            "address": "Tonk Road, Near SMS Stadium, Jaipur, Rajasthan 302015",
            "contact": "+91 (141) 666-7777",
            "timing": "Mon–Sat: 8 AM – 8 PM, Sun: 10 AM – 6 PM"
        },
        {
            "name": "🏥 Max Healthcare - Dermatology",
            "address": "Jagatpura, Jaipur, Rajasthan 302030",
            "contact": "+91 (141) 445-5555",
            "timing": "Mon–Fri: 9 AM – 6 PM, Sat: 9 AM – 5 PM, Sun: Closed"
        },
        {
            "name": "🏥 Glowderma Skin & Hair Clinic",
            "address": "Bani Park, Jaipur, Rajasthan 302016",
            "contact": "+91 (141) 234-5678",
            "timing": "Tue–Sun: 10 AM – 7 PM, Mon: Closed"
        },
        {
            "name": "🏥 Fortis Hospital - Dermatology Wing",
            "address": "Chankyapuri, Jaipur, Rajasthan 302021",
            "contact": "+91 (141) 888-9999",
            "timing": "24/7 Emergency, Appointments: 8 AM – 8 PM"
        }
    ]
    return render_template("clinics.html", clinics=clinics_data)

# AI PREDICTION & REPORTS
@app.route('/predict', methods=['POST'])
def predict():
    """Handles skin disease prediction using trained deep learning model"""

    if 'file' not in request.files:
        return redirect(url_for('home'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('home'))

    # Save uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # ------------------------------
    # 1️⃣ Read + Preprocess Image
    # ------------------------------
    img = cv2.imread(filepath)
    if img is None:
        return "Error: Unable to read uploaded image."

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # ------------------------------
    # 2️⃣ Predict
    # ------------------------------
    pred = model.predict(img)
    pred = pred[0]

    top_2 = np.argsort(pred)[-2:][::-1]
    pred_class_1st = top_2[0]
    pred_class_2nd = top_2[1]

    confidence_1st = pred[pred_class_1st]
    confidence_2nd = pred[pred_class_2nd]

    MIN_CONFIDENCE = 0.10
    CONFIDENCE_GAP = 0.15

    # Map label
    disease_1st = label_map[pred_class_1st]
    short_1st = disease_1st.split()[0]

    # ------------------------------
    # 3️⃣ Confidence Logic
    # ------------------------------
    if confidence_1st < MIN_CONFIDENCE:
        final_prediction = "⚠️ Uncertain — confidence too low"
        final_short = "uncertain"
        final_conf = confidence_1st

    elif (confidence_1st - confidence_2nd) < CONFIDENCE_GAP:
        final_prediction = f"Ambiguous: {label_map[pred_class_1st]} OR {label_map[pred_class_2nd]}"
        final_short = "ambiguous"
        final_conf = confidence_1st

    else:
        final_prediction = disease_1st
        final_short = short_1st
        final_conf = confidence_1st

    # ------------------------------
    # 4️⃣ Save Report to Database
    # ------------------------------
    current_user = get_current_user()
    user_name = current_user.username if current_user else "Guest User"

    new_report = Report(
        user_name=user_name,
        disease=final_prediction,
        confidence=round(final_conf * 100, 2),
        image_path=filename
    )
    db.session.add(new_report)
    db.session.commit()

    # ------------------------------
    # 5️⃣ Update History Log
    # ------------------------------
    history.append({
        "image": filename,
        "disease": final_prediction,
        "confidence": round(final_conf * 100, 2),
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    })

    # ------------------------------
    # 6️⃣ Render Result Page
    # ------------------------------
    return render_template(
        'result.html',
        filename=filename,
        disease=final_prediction,
        confidence=round(final_conf * 100, 2),
        desc=info.get(final_short, {}).get("desc", "Consult a dermatologist for proper diagnosis."),
        tip=info.get(final_short, {}).get("tip", "Visit a skin specialist for accurate confirmation."),
        history=history[-5:]
    )

@app.route("/reports")
def reports():
    """
    Shows all reports
    """
    all_reports = Report.query.order_by(Report.created_at.desc()).all()
    return render_template("reports.html", reports=all_reports)

@app.route('/report/<filename>/<disease>/<confidence>')
def report(filename, disease, confidence):
    """
    Generates PDF report
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="AI Skin Disease Detection Report", ln=1, align='C')
    pdf.cell(200, 10, txt=f"Disease: {disease}", ln=2)
    pdf.cell(200, 10, txt=f"Confidence: {confidence}%", ln=3)
    pdf.output("static/uploads/report.pdf")
    return "Report saved as static/uploads/report.pdf"

# CHATBOT & FEEDBACK

OPENROUTER_API_KEY = "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"

@app.route("/chat", methods=["POST"])
def chat():
    """
    AI chatbot for skincare advice
    """
    try:
        user_message = request.json.get("message", "").strip()
        if not user_message:
            return jsonify({"reply": "Please type a message first 😊"})

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "gryphe/mythomax-l2-13b",
            "messages": [
                {"role": "system", "content": "You are a friendly AI assistant that helps users with skin disease detection, skincare advice, and understanding reports."},
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.6,
            "max_tokens": 400
        }

        resp = requests.post(url, headers=headers, json=data)
        resp_json = resp.json()
        bot_reply = resp_json.get("choices", [{}])[0].get("message", {}).get("content", "")

        bot_reply = re.sub(r'\[.*?\]', '', bot_reply)
        bot_reply = bot_reply.replace('**', '').replace('__', '')
        bot_reply = bot_reply.replace('. ', '.<br><br>')

        if not bot_reply.strip():
            bot_reply = "🤖 Sorry, I didn't quite catch that — please try again!"

        return jsonify({"reply": bot_reply})

    except Exception as e:
        print("❌ Chat Error:", e)
        return jsonify({"reply": f"Server error: {e}"})

@app.route("/feedback", methods=["GET", "POST", "OPTIONS"])
def feedback():
    print(f"🔍 REQUEST RECEIVED!")
    print(f"📋 Method: {request.method}")
    print(f"📋 Content-Type: {request.content_type}")
    print(f"📋 Is JSON: {request.is_json}")
    print(f"📋 URL: {request.url}")
    
    if request.method == "OPTIONS":
        print("✅ OPTIONS request - CORS preflight")
        return "", 204
    
    if request.method == "GET":
        print("❌ GET request received (should be POST!)")
        return jsonify({"message": "Send feedback via POST"}), 200
    
    if request.method == "POST":
        try:
            print("✅ POST request received")
            
            if request.is_json:
                data = request.get_json()
                print(f"✅ JSON data received: {data}")
            else:
                data = request.form.to_dict()
                print(f"✅ Form data received: {data}")
            
            name = data.get("name", "").strip()
            email = data.get("email", "").strip()
            message = data.get("message", "").strip()
            feedback_type = data.get("feedback_type", "").strip()
            device_type = data.get("device_type", "").strip()
            rating = data.get("rating")
            page = data.get("page", "").strip()
            consent = data.get("agree") or data.get("consent")

            print(f"📝 Parsed data - Name: {name}, Email: {email}, Type: {feedback_type}")

            if not all([name, email, message]):
                print("⚠️ Missing required fields")
                return jsonify({
                    "success": False,
                    "reply": "⚠️ Please fill in all required fields!"
                }), 400

            new_feedback = Feedback(
                name=name,
                email=email,
                message=message,
                feedback_type=feedback_type or None,
                device_type=device_type or None,
                rating=int(rating) if rating else None,
                page=page or None,
                consent=bool(consent)
            )
            db.session.add(new_feedback)
            db.session.commit()

            print(f"✅ Feedback saved successfully from {name}")
            return jsonify({
                "success": True,
                "reply": "✅ Thank you for your feedback!"
            }), 200

        except Exception as e:
            db.session.rollback()
            print(f"❌ Feedback Error: {e}")
            return jsonify({
                "success": False,
                "reply": f"Error saving feedback: {e}"
            }), 500

# RUN APP

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        print("✅ Database created successfully!")
        print("🔐 Auth routes: /signup, /login, /logout, /check-auth")
        print("🌐 Server running on http://localhost:5001")
    app.run(debug=True, use_reloader=False, port=5001)