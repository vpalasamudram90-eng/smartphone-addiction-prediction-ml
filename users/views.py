import os
from django.conf import settings
from django.shortcuts import render
import joblib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE 

def Training(request):

    import pandas as pd
    import joblib
    import seaborn as sns
    import matplotlib.pyplot as plt

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.ensemble import RandomForestClassifier

    # ---------------------------
    # Load dataset
    # ---------------------------
    df = pd.read_csv('media/ClassSurvey.csv')

    # ---------------------------
    # Encode target
    # ---------------------------
    target_encoder = LabelEncoder()
    df['SocialMediaAddiction'] = target_encoder.fit_transform(
        df['SocialMediaAddiction']
    )

    # ---------------------------
    # SELECT 15 NUMERIC FEATURES
    # ---------------------------
    features = [
        'Age',
        'Whatsapp',
        'Instagram',
        'Snapchat',
        'Telegram',
        'Facebook',
        'TikTok',
        'WeChat',
        'Twitter',
        'Linkedin',
        'Messages',
        'TotalSocialMediaScreenTime',
        'Number of times opened (hourly intervals)',
        'Daily Usage Hours',
        'BeReal'
    ]

    target = 'SocialMediaAddiction'

    # ---------------------------
    # Force numeric conversion
    # ---------------------------
    for col in features:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # ---------------------------
    # Handle missing values
    # ---------------------------
    imputer = SimpleImputer(strategy='mean')
    df[features] = imputer.fit_transform(df[features])

    # ---------------------------
    # Feature / Target split
    # ---------------------------
    X = df[features]
    y = df[target]

    # ---------------------------
    # Train / Test split (NO SMOTE)
    # ---------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=39,
        stratify=y
    )

    # ---------------------------
    # Train model
    # ---------------------------
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
    model.fit(X_train, y_train)

    # ---------------------------
    # Evaluation
    # ---------------------------
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig('media/confusion_matrix.png')
    plt.close()

    # Feature Importance
    feature_df = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(8,6))
    sns.barplot(x='Importance', y='Feature', data=feature_df)
    plt.title('Feature Importance')
    plt.savefig('media/feature_importance.png')
    plt.close()

    # ---------------------------
    # Save model
    # ---------------------------
    joblib.dump(model, 'media/social_media_addiction_model.pkl')

    return render(request, 'users/accuracy.html', {
        'accuracy': accuracy,
        'confusion_matrix_img': 'media/confusion_matrix.png',
        'feature_importance_img': 'media/feature_importance.png'
    })

# def Prediction(request):
#     if request.method == "POST":

#         import numpy as np
#         import joblib

#         # ---------------------------
#         # Load trained model
#         # ---------------------------
#         model = joblib.load('media/social_media_addiction_model.pkl')

#         # ---------------------------
#         # Read ONLY numeric inputs
#         # (same order as training)
#         # ---------------------------
#         age = float(request.POST.get('age'))
#         whatsapp = float(request.POST.get('whatsapp'))
#         instagram = float(request.POST.get('instagram'))
#         snapchat = float(request.POST.get('snapchat'))
#         telegram = float(request.POST.get('telegram'))
#         facebook = float(request.POST.get('facebook'))
#         tiktok = float(request.POST.get('tiktok'))
#         wechat = float(request.POST.get('wechat'))
#         twitter = float(request.POST.get('twitter'))
#         linkedin = float(request.POST.get('linkedin'))
#         messages = float(request.POST.get('messages'))
#         total_time = float(request.POST.get('total_time'))

#         bereal = float(request.POST.get('bereal'))

#         # ---------------------------
#         # Create model input
#         # EXACT ORDER as training
#         # ---------------------------
#         input_data = np.array([[
#             age,
#             whatsapp,
#             instagram,
#             snapchat,
#             telegram,
#             facebook,
#             tiktok,
#             wechat,
#             twitter,
#             linkedin,
#             messages,
#             total_time,

#             bereal
#         ]])

#         # ---------------------------
#         # Prediction
#         # ---------------------------
#         prediction = model.predict(input_data)
#         result = "Addicted" if prediction[0] == 1 else "Not Addicted"

#         # ---------------------------
#         # Render result
#         # ---------------------------
#         return render(request, 'users/detection.html', {
#             'result': result,
#             'age': age,
#             'whatsapp': whatsapp,
#             'instagram': instagram,
#             'snapchat': snapchat,
#             'telegram': telegram,
#             'facebook': facebook,
#             'tiktok': tiktok,
#             'wechat': wechat,
#             'twitter': twitter,
#             'linkedin': linkedin,
#             'messages': messages,
#             'total_time': total_time,

#             'bereal': bereal,
#         })

#     return render(request, 'users/test_input.html')





from django.shortcuts import render
import pandas as pd
import joblib
import os

# -----------------------------------------------------
# Load model
# -----------------------------------------------------
MODEL_PATH = os.path.join("media", "models", "smartphone_addiction_model.pkl")

data = joblib.load(MODEL_PATH)

model = data["model"]
scaler = data["scaler"]
feature_names = data["feature_names"]

# -----------------------------------------------------
# Prediction View
# -----------------------------------------------------
def predict_addiction(request):
    prediction = None

    if request.method == "POST":
        user_input = {}

        # Collect dropdown values
        for feature in feature_names:
            value = request.POST.get(feature)

            # Safety: empty -> 0
            if value is None or value == "":
                value = 0

            user_input[feature] = int(value)

        # Create DataFrame
        df = pd.DataFrame([user_input])

        # Ensure column order
        df = df[feature_names]

        # Scale
        df_scaled = scaler.transform(df)

        # Predict
        result = int(model.predict(df_scaled)[0])

        if result == 0:
            prediction = "Not Addicted ✅"
        elif result == 1:
            prediction = "Risk of Addiction ⚠️"
        else:
            prediction = "Addicted 📵"

    return render(
        request,
        "users/predict.html",
        {
            "prediction": prediction,
            "features": feature_names
        }
    )




def ViewDataset(request):
    # Path to your actual dataset
    dataset_path = os.path.join(settings.MEDIA_ROOT, "final dataset.csv")

    # Read Excel file (limit rows for performance)
    df = pd.read_csv(dataset_path, nrows=100)

    # Convert DataFrame to HTML table
    table = df.to_html(
        classes="table table-striped table-bordered",
        index=False
    )

    return render(
        request,
        "users/viewData.html",
        {"data": table}
    )


from django.shortcuts import render, redirect
from .models import UserRegistrationModel
from django.contrib import messages

def UserRegisterActions(request):
    if request.method == 'POST':
        user = UserRegistrationModel(
            name=request.POST['name'],
            loginid=request.POST['loginid'],
            password=request.POST['password'],
            mobile=request.POST['mobile'],
            email=request.POST['email'],
            locality=request.POST['locality'],
            address=request.POST['address'],
            city=request.POST['city'],
            state=request.POST['state'],
            status='waiting'
        )
        user.save()
        messages.success(request,"Registration successful!")
    return render(request, 'UserRegistrations.html') 


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                data = {'loginid': loginid}
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})

def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})


def index(request):
    return render(request,"index.html")


import random
from django.shortcuts import render, redirect
from django.core.mail import send_mail
from django.contrib import messages
from .models import UserRegistrationModel

otp_storage = {}  # Temporary dictionary to store OTPs

def send_otp(email):
    otp = random.randint(100000, 999999)  # Generate a 6-digit OTP
    otp_storage[email] = otp

    subject = "Password Reset OTP"
    message = f"Your OTP for password reset is: {otp}"
    from_email = "saikumardatapoint1@gmail.com"  # Change this to your email
    send_mail(subject, message, from_email, [email])

    return otp

def forgot_password(request):
    if request.method == "POST":
        email = request.POST.get("email")

        if UserRegistrationModel.objects.filter(email=email).exists():
            send_otp(email)
            request.session["reset_email"] = email  # Store email in session
            return redirect("verify_otp")
        else:
            messages.error(request, "Email not registered!")

    return render(request, "users/forgot_password.html")

def verify_otp(request):
    if request.method == "POST":
        otp_entered = request.POST.get("otp")
        email = request.session.get("reset_email")

        if otp_storage.get(email) and str(otp_storage[email]) == otp_entered:
            return redirect("reset_password")
        else:
            messages.error(request, "Invalid OTP!")

    return render(request, "users/verify_otp.html")

def reset_password(request):
    if request.method == "POST":
        new_password = request.POST.get("new_password")
        email = request.session.get("reset_email")

        if UserRegistrationModel.objects.filter(email=email).exists():
            user = UserRegistrationModel.objects.get(email=email)
            user.password = new_password  # Updating password
            user.save()
            messages.success(request, "Password reset successful! Please log in.")
            return redirect("UserLoginCheck")

    return render(request, "users/reset_password.html")



import os
import joblib
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from django.conf import settings
from django.shortcuts import render
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def model_evaluation(request):

    # --------------------------------------------------
    # 1. Load trained model
    # --------------------------------------------------
    MODEL_PATH = os.path.join(
        settings.MEDIA_ROOT,
        "models",
        "smartphone_addiction_model.pkl"
    )

    model_data = joblib.load(MODEL_PATH)
    model = model_data["model"]
    scaler = model_data["scaler"]
    feature_names = model_data["feature_names"]

    # --------------------------------------------------
    # 2. Load dataset
    # --------------------------------------------------
    DATASET_PATH = os.path.join(
        settings.MEDIA_ROOT,
        "final dataset.csv"
    )

    df = pd.read_csv(DATASET_PATH)

    y = df["target"]
    X = df.drop(columns=["target"])

    # --------------------------------------------------
    # 3. Preprocessing (same as training)
    # --------------------------------------------------
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())
    X = X[feature_names]

    X_scaled = scaler.transform(X)

    # --------------------------------------------------
    # 4. Prediction
    # --------------------------------------------------
    y_pred = model.predict(X_scaled)

    accuracy = round(accuracy_score(y, y_pred) * 100, 2)

    # --------------------------------------------------
    # 5. Classification Report
    # --------------------------------------------------
    report = classification_report(
        y,
        y_pred,
        target_names=[
            "Not Addicted",
            "Risk of Addiction",
            "Addicted"
        ],
        output_dict=True,
        zero_division=0
    )

    # 🔧 FIX: rename "f1-score" → "f1_score" (Django-safe)
    for label in report:
        if isinstance(report[label], dict) and "f1-score" in report[label]:
            report[label]["f1_score"] = report[label].pop("f1-score")

    # --------------------------------------------------
    # 6. Confusion Matrix
    # --------------------------------------------------
    cm = confusion_matrix(y, y_pred)

    os.makedirs("static/plots", exist_ok=True)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Purples",
        xticklabels=["Not Addicted", "Risk", "Addicted"],
        yticklabels=["Not Addicted", "Risk", "Addicted"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("static/plots/confusion_matrix.png")
    plt.close()

    # --------------------------------------------------
    # 7. Feature Importance Plot
    # --------------------------------------------------
    importances = model.feature_importances_

    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    plt.figure(figsize=(8, 6))
    sns.barplot(
        x="importance",
        y="feature",
        data=fi_df,
        palette="viridis"
    )
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig("static/plots/feature_importance.png")
    plt.close()

    # --------------------------------------------------
    # 8. Render page
    # --------------------------------------------------
    return render(
        request,
        "users/model_evaluation.html",
        {
            "accuracy": accuracy,
            "confusion_matrix": "plots/confusion_matrix.png",
            "feature_plot": "plots/feature_importance.png",
            "report": report
        }
    )