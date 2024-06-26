from flask import Flask, render_template, request, session, redirect, url_for
import mysql.connector
import pickle

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Connect to MySQL database
db_connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="hospital"
)
cursor = db_connection.cursor()

# Load the models (Make sure you have these files in your directory or update paths)
heart_model = pickle.load(open('heart-disease-prediction-model.pkl', 'rb'))
diabetes_model = pickle.load(open('diabetes-prediction-model.pkl', 'rb'))
knn_heart_model = pickle.load(open('knn-model-heart.pkl', 'rb'))
knn_diabetes_model = pickle.load(open('knn-model-diabetes.pkl', 'rb'))

@app.route('/')
def home():
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role = request.form['role']
        
        # Check if the user exists in the database and has the correct role
        cursor.execute("SELECT * FROM users WHERE username = %s AND password = %s AND role = %s", (username, password, role))
        user = cursor.fetchone()
        
        if user:
            session['logged_in'] = True
            session['role'] = role
            session['username'] = username  # Store username in session
            
            success_message = 'Successfully logged in!'
            
            if role == 'doctor':
                return redirect(url_for('doctor_dashboard', success_message=success_message))
            elif role == 'patient':
                return redirect(url_for('patient_dashboard', success_message=success_message))
        else:
            error_message = 'Invalid username, password, or role.'
            return render_template('login.html', message=error_message)
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role = request.form['role']
        
        # Check if the username already exists
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        
        if user:
            return render_template('signup.html', message='Username already exists.')
        else:
            # Insert the new user into the database
            cursor.execute("INSERT INTO users (username, password, role) VALUES (%s, %s, %s)", (username, password, role))
            db_connection.commit()
            return redirect(url_for('login'))
    
    return render_template('signup.html')

@app.route('/doctor_dashboard')
def doctor_dashboard():
    if 'logged_in' in session and session['role'] == 'doctor':
        return render_template('doctor_dashboard.html')
    else:
        return redirect(url_for('home'))

@app.route('/patient_dashboard')
def patient_dashboard():
    if 'logged_in' in session and session['role'] == 'patient':
        return render_template('patient_dashboard.html')
    else:
        return redirect(url_for('home'))
    
@app.route('/attribute_info')
def attribute_info():
    return render_template('attribute_info.html')

@app.route('/heart_attribute_info')
def heart_attribute_info():
    return render_template('heart_attribute_info.html')


@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('role', None)
    return redirect(url_for('home'))

@app.route('/heart_disease_prediction', methods=['GET', 'POST'])
def heart_disease_prediction():
    if request.method == 'POST':
        data = [float(x) for x in request.form.values()]
        rf_prediction_proba = heart_model.predict_proba([data])[0][1]
        knn_prediction_proba = knn_heart_model.predict_proba([data])[0][1]
        combined_proba = (rf_prediction_proba + knn_prediction_proba) / 2

        attributes = ['Age', 'Sex', 'Chest Pain Type', 'Resting Blood Pressure', 'Cholesterol', 'Fasting Blood Sugar', 'Resting ECG', 'Max Heart Rate', 'Exercise Induced Angina', 'ST Depression', 'Slope of ST Segment', 'Number of Major Vessels', 'Thalassemia']
        values = list(request.form.values())

        return render_template('result.html', combined_proba=combined_proba,
                               attributes=attributes, values=values)
    return render_template('main.html')

@app.route('/diabetes_prediction', methods=['GET', 'POST'])
def diabetes_prediction():
    if request.method == 'POST':
        data = [float(x) if i in [5, 6] else int(x) for i, x in enumerate(request.form.values())]
        rf_prediction_proba = diabetes_model.predict_proba([data])[0][1]
        knn_prediction_proba = knn_diabetes_model.predict_proba([data])[0][1]
        diabetes_prediction_proba = (rf_prediction_proba + knn_prediction_proba) / 2

        attributes = ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin', 'BMI', 'Diabetes Pedigree Function', 'Age']
        values = list(request.form.values())

        return render_template('diabetes_result.html', prediction_type='Diabetes',
                               rf_prediction_proba=rf_prediction_proba, knn_prediction_proba=knn_prediction_proba,
                               diabetes_prediction_proba=diabetes_prediction_proba,
                               attributes=attributes, values=values)
    return render_template('diabetes.html', prediction_type='diabetes')

@app.route('/diabetes_remedies')
def diabetes_remedies():
    return render_template('diabetes_remedies.html')

if __name__ == '__main__':
    app.run(debug=True)
