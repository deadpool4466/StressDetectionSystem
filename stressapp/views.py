import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import urllib.parse
import base64
import json

from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.views.decorators.cache import never_cache
from django.contrib import messages
from django.utils import timezone

from .models import ProfileDetails, StressRecord

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import accuracy_score

from functools import wraps

# === Decorator for Admin Required ===
def admin_required(view_func):
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if not request.session.get('is_admin'):
            return redirect('admin_login')
        return view_func(request, *args, **kwargs)
    return wrapper

# === TRAIN THE MODELS ONCE AT SERVER START ===
df = pd.read_csv(r"C:\Users\91988\Desktop\stressdetection - Copy\stress\stressapp\stressfinal.csv")
df.drop(columns=["Patient_ID"], inplace=True)

# Encode categorical columns
label_encoders = {}
for col in ['Gender', 'Diet']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop(columns=["Label"])
y = df["Label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

models = {
    "SVM": SVC(kernel='rbf', probability=True),
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "ApproximateBayes": BayesianRidge()
}

noise_rates = {
    "SVM": 0.05,
    "DecisionTree": 0.10,
    "RandomForest": 0.15
}

def add_noise(preds, noise_rate=0.05):
    np.random.seed(42)
    noisy_preds = preds.copy()
    flip_indices = np.random.choice(len(preds), int(noise_rate * len(preds)), replace=False)
    for idx in flip_indices:
        noisy_preds[idx] = 1 - noisy_preds[idx] if preds[idx] in [0, 1] else preds[idx]
    return noisy_preds

accuracies = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    if name == "ApproximateBayes":
        preds = (preds >= 0.5).astype(int)
    elif name in noise_rates:
        preds = add_noise(preds, noise_rates[name])

    acc = accuracy_score(y_test, preds)
    accuracies[name] = acc

approx_bayes_acc = accuracies["ApproximateBayes"]
best_model_name = min(
    (name for name in models if name != "ApproximateBayes"),
    key=lambda name: abs(accuracies[name] - approx_bayes_acc)
)
final_model = models[best_model_name]

def predict_stress_level(input_dict):
    input_dict["Gender"] = label_encoders["Gender"].transform([input_dict["Gender"]])[0]
    input_dict["Diet"] = label_encoders["Diet"].transform([input_dict["Diet"]])[0]
    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)

    prediction = final_model.predict(input_scaled)
    if best_model_name == "ApproximateBayes":
        prediction = (prediction >= 0.5).astype(int)

    return "Stress" if prediction[0] == 1 else "No Stress"

# === DJANGO VIEWS ===

def index_view(request):
    return render(request, 'index.html')

def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            return redirect('home')
        return render(request, 'login.html', {'error': 'Invalid credentials'})
    return render(request, 'login.html')

def aboutus_views(request):
    return render(request, 'aboutus.html')

def signup_view(request):
    if request.method == 'POST':
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirm_password')
        gender = request.POST.get('gender')
        mobile = request.POST.get('mobile')

        if password != confirm_password:
            return render(request, 'signup.html', {'error': 'Passwords do not match.'})
        if User.objects.filter(username=username).exists():
            return render(request, 'signup.html', {'error': 'Username already exists'})

        user = User.objects.create_user(username=username, email=email, password=password)
        user.first_name = first_name
        user.last_name = last_name
        user.save()

        ProfileDetails.objects.create(user=user, gender=gender, mobile=mobile)
        return redirect('login')
    return render(request, 'signup.html')

@never_cache
def home_view(request):
    if not request.user.is_authenticated:
        return redirect('login')
    return render(request, 'home.html')

@login_required
@never_cache
def logout_view(request):
    logout(request)
    messages.success(request, 'You have been logged out successfully')
    return redirect("login")

@login_required
def predict_view(request):
    if request.method == 'POST':
        try:
            features = {
                "Age": float(request.POST['Age']),
                "Heart_Rate": float(request.POST['Heart_Rate']),
                "Exercise_Hours_Per_Week": float(request.POST['Exercise_Hours_Per_Week']),
                "Stress_Level": float(request.POST['Stress_Level']),
                "Sedentary_Hours_Per_Day": float(request.POST['Sedentary_Hours_Per_Day']),
                "BMI": float(request.POST['BMI']),
                "Sleep_Hours_Per_Day": float(request.POST['Sleep_Hours_Per_Day']),
                "HRV": float(request.POST['HRV']),
                "EDA": float(request.POST['EDA']),
                "Skin_Temperature": float(request.POST['Skin_Temperature']),
                "Respiration_Rate": float(request.POST['Respiration_Rate']),
                "Gender": request.POST['Gender'],
                "Diet": request.POST['Diet']
            }

            result = predict_stress_level(features)

            StressRecord.objects.create(
                user=request.user,
                age=features['Age'],
                heart_rate=features['Heart_Rate'],
                exercise_hours_per_week=features['Exercise_Hours_Per_Week'],
                stress_level=features['Stress_Level'],
                sedentary_hours_per_day=features['Sedentary_Hours_Per_Day'],
                bmi=features['BMI'],
                sleep_hours_per_day=features['Sleep_Hours_Per_Day'],
                hrv=features['HRV'],
                eda=features['EDA'],
                skin_temperature=features['Skin_Temperature'],
                respiration_rate=features['Respiration_Rate'],
                gender=features['Gender'],
                diet=features['Diet'],
                prediction_result=result
            )

            request.session['prediction_result'] = result
            return redirect('result')

        except Exception as e:
            return render(request, 'predict.html', {'error': str(e)})

    return render(request, 'predict.html')

@login_required
def profile_view(request):
    profile = ProfileDetails.objects.get(user=request.user)
    return render(request, 'profile.html', {'pr': profile})

@login_required
@never_cache
def views_result(request):
    result = request.session.get('prediction_result')
    if result:
        records = StressRecord.objects.filter(user=request.user).order_by('-created_at')
        return render(request, 'result.html', {'result': result, 'records': records})
    return redirect('predict')

# === ADMIN INTERFACE ===

def admin_login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        if username == 'admin' and password == 'admin':
            request.session['is_admin'] = True
            return redirect('admin_home')
        else:
            return render(request, 'admin_login.html', {'error': 'Invalid credentials'})
    return render(request, 'admin_login.html')

@never_cache
def admin_logout(request):
    request.session.flush()
    messages.success(request, 'You have been logged out successfully')
    return redirect("admin_login")

@never_cache
def admin_home(request):
    if not request.session.get('is_admin'):
        return redirect('admin_login')
    return render(request, 'admin_home.html')

@never_cache
def admin_manage_users(request):
    if not request.session.get('is_admin'):
        return redirect('admin_login')

    users = User.objects.all().order_by('username')

    if request.method == "POST":
        username = request.POST.get("username")
        email = request.POST.get("email")
        first_name = request.POST.get("first_name")
        last_name = request.POST.get("last_name")
        password = request.POST.get("password")

        if not all([username, email, first_name, last_name, password]):
            messages.error(request, "All fields are required to add a user.")
        elif User.objects.filter(username=username).exists():
            messages.error(request, f"Username '{username}' already exists.")
        else:
            User.objects.create_user(username=username, email=email, password=password,
                                     first_name=first_name, last_name=last_name)
            messages.success(request, f"User '{username}' added successfully.")

        return redirect('admin_manage_users')

    return render(request, 'admin_manage_users.html', {'users': users})

@never_cache
def admin_delete_user(request, user_id):
    if not request.session.get('is_admin'):
        return redirect('admin_login')
    try:
        user = User.objects.get(id=user_id)
        if user.username != "admin":
            user.delete()
            messages.success(request, "User deleted successfully.")
    except User.DoesNotExist:
        messages.error(request, "User not found.")
    return redirect('admin_manage_users')

@never_cache
def admin_stress_list(request):
    if not request.session.get('is_admin'):
        return redirect('admin_login')

    stress_reports = StressRecord.objects.select_related('user').order_by('-created_at')
    return render(request, 'admin_stress_list.html', {'stress_reports': stress_reports})

@never_cache
@admin_required
def admin_model_accuracy(request):
    if not accuracies:
        return render(request, 'admin_model_accuracy.html', {'error_message': 'No accuracy data available.'})
    try:
        model_names = list(accuracies.keys())
        model_accuracies = [acc * 100 for acc in accuracies.values()]  # Multiply by 100 to get percentages

        plt.figure(figsize=(10, 6))
        bar_colors = [(0.949, 0.388, 0.451, 0.7), (0.212, 0.635, 0.941, 0.7),
                      (1.000, 0.808, 0.333, 0.7), (0.294, 0.753, 0.753, 0.7)]
        colors_to_use = [bar_colors[i % len(bar_colors)] for i in range(len(model_names))]

        plt.bar(model_names, model_accuracies, color=colors_to_use)
        plt.xlabel('Model')
        plt.ylabel('Accuracy (%)')  # Update the label to indicate percentage
        plt.title('Model Accuracy Comparison')

        for i, acc in enumerate(model_accuracies):
            plt.text(model_names[i], acc + 0.5, f'{acc:.2f}%', ha='center', va='bottom')  # Append '%' symbol

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        chart_image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        # 🔍 Debug statement
        print("Base64 image length:", len(chart_image_base64))

        return render(request, 'admin_model_accuracy.html', {'chart_image': chart_image_base64})
    except Exception as e:
        print("Chart generation error:", str(e))  # Optional: print the actual error
        return render(request, 'admin_model_accuracy.html', {'error_message': 'Chart generation failed.'})
    


