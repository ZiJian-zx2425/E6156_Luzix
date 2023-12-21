import pathlib

# import pymongo
# from database import *

import cachecontrol
from flask import Flask, session, abort, redirect, request, render_template, url_for, request, jsonify
from flask_cors import CORS, cross_origin
import os
import requests

from google.oauth2 import id_token
from google_auth_oauthlib.flow import Flow
import google.auth.transport.requests
import jwt
import datetime

import asyncio
import aiohttp
import random

app = Flask(__name__)
CORS(app)
app_name = app.name


# print(f"The name of the Flask application is: {app_name}")
app.secret_key = "CodeSpecialist.com"
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
GOOGLE_CLIENT_ID = "665838449619-o7a03bq8ievn06qrs7vplvf5ovc3gu47.apps.googleusercontent.com"
client_secrets_file = os.path.join(pathlib.Path(__file__).parent, "client_secret.json")


users_db = {
    '110788464327696265201': 'patient',
    '104405107080836112407': 'doctor',
    '117740487543455173970': 'volunteer',
}


flow = Flow.from_client_secrets_file(
    client_secrets_file=client_secrets_file,
    scopes=["https://www.googleapis.com/auth/userinfo.profile", "https://www.googleapis.com/auth/userinfo.email", "openid"],
    redirect_uri="http://127.0.0.1:5001/callback"
)


def login_is_required(function):
    def wrapper(*args, **kwargs):
        if "google_id" not in session:
            return abort(401)  # Authorization required
        else:
            return function()
    return wrapper


# map doctor * patients
# google_id * [a list of patient ids e.x. (1, 3)]

# test
# doctor 1: shuting.li.sli2@gmail.com
# doctor 2:

# modify the get patient_record http get patient_record method to get by patient id


doctors_office_hours = {
    "Dr. John Doe": ["2023-04-01T09:00", "2023-04-01T10:00"],
    "Dr. Jane Smith": ["2023-04-02T11:00", "2023-04-02T12:00"]
}

# doctor_patient_relationship = {google_id, patient name}
doctors_patients = {
    '104405107080836112407': ['Super Mario', 'Princess Peach']
}

appointments = [
    {
        "id": 1,
        "patientName": "John Doe",
        "doctorName": "Dr. Smith",
        "dateTime": "2023-12-01T10:00",
        "status": "pending"
    },
    {
        "id": 2,
        "patientName": "Jane Smith",
        "doctorName": "Dr. Jones",
        "dateTime": "2023-12-02T15:00",
        "status": "pending"
    },
    {
        "id": 3,
        "patientName": "Jodi",
        "doctorName": "Matt Candice",
        "dateTime": "2023-12-02T15:00",
        "status": "pending"
    }
]

patients_records = [
    {'id': 1, 'name': 'Super Mario', 'condition': 'headache'},
    {'id': 2, 'name': 'Luigi', 'condition': 'covid'},
    {'id': 3, 'name': 'Princess Peach', 'condition': 'allergies'},
    {'id': 4, 'name': 'King Boo', 'condition': 'flu'},
]

@app.route("/")
def navbar():
    print("main page accessed")
    print(session)
    if 'google_id' not in session:
        # 用户未登录，重定向到登录页面
        return redirect('/login')
    user_role = session.get('role', 'not logged in')
    return "hey."

# SSO Authentication Logic
@app.route("/login")
def login():
    print("login page accessed")
    authorization_url, state = flow.authorization_url(
        prompt='consent'  # Forces re-consent and re-authentication
    )
    print("authorization_url: ", authorization_url)
    session["state"] = state
    return redirect(authorization_url)  # TODO: change another redirection url

@app.route("/callback")
def callback():
    print("callback endpoint is accessed")
    flow.fetch_token(authorization_response=request.url)

    if not session["state"] == request.args["state"]:
        abort(500)

    credentials = flow.credentials
    request_session = requests.session()
    cached_session = cachecontrol.CacheControl(request_session)
    token_request = google.auth.transport.requests.Request(session=cached_session)

    # verify with google?
    id_info = id_token.verify_oauth2_token(
        id_token=credentials._id_token,
        request=token_request,
        audience=GOOGLE_CLIENT_ID
    )
    # set session data
    session["google_id"] = id_info.get("sub")
    session["name"] = id_info.get("name")
    session['role'] = users_db.get(id_info.get("sub"), 'not logged in') # set user role
    token = create_token(id_info.get("sub"), id_info.get("name"), users_db.get(id_info.get("sub"), 'not logged in'))
    return redirect(f"http://localhost:4200/auth-success?token={token}")

    # return redirect("/")
    # return redirect("http://localhost:4200/schedule")
    # return redirect("http://localhost:4200/appointments")
    # return redirect("/appointments")


def create_token(user_id, user_name, user_role):
    payload = {
        'exp': datetime.datetime.utcnow() + datetime.timedelta(days=1),  # Token expiration time
        'iat': datetime.datetime.utcnow(),  # Issued at time
        'sub': user_id,
        'name': user_name,
        'role': user_role
    }
    return jwt.encode(payload, 'your_secret_key', algorithm='HS256')


@app.route("/logout")
def logout():
    print("logout endpoint accessed")
    # Optional: Revoke the Google token
    if 'credentials' in session:
        credentials = google.oauth2.credentials.Credentials(
            **session['credentials'])
        revoke = request.Request(
            'https://accounts.google.com/o/oauth2/revoke',
            params={'token': credentials.token},
            headers={'content-type': 'application/x-www-form-urlencoded'})
        try:
            request.urlopen(revoke)
        except Exception as e:
            print(f'An error occurred: {e}')
    session.clear()
    return redirect("/appointments")    #TODO: change another redirection url


@app.route('/role')
def get_role():
    user_role = session.get('role', 'not logged in')
    return jsonify({"role": user_role})

'''
# -------------------------patient page---------------------------
@app.route('/basic_info')
def get_basic():
    gid = session.get('google_id')
    # return jsonify({"role": user_role})
    user_data = db_search(collection, users_db[gid][0])
    user_data['health'] = db_count(collection, users_db[gid][0])
    if user_data:
        return jsonify(user_data)
    else:
        return jsonify({"error": "Record not found"}), 404


@app.route('/history/<date>')
def get_record(date):
    gid = session.get('google_id')
    user_data = daily_record(collection, users_db[gid][0], date)
    print(user_data)
    if user_data:
        return jsonify(user_data)
    else:
        return jsonify({"error": "Record not found"}), 404


@app.route('/history/submit-current-info', methods=['POST'])
def submit_current_info():
    data = request.json
    gid = session.get('google_id')
    data['pid'] = users_db[gid][0]
    data['time'] = datetime.now().date().strftime("%Y-%m-%d")
    update_record(collection, data)
    # Process and store the data in your database
    # For example: add_record_to_database(data)
    return jsonify({"message": "Data submitted successfully"})

@app.route("/labs")
def labs():
    return render_template("labs.html")

'''

@app.route('/api/is_logged_in')
def is_logged_in():
    is_logged_in = 'google_id' in session
    user_role = session.get('role', 'not logged in') if is_logged_in else 'not logged in'
    return jsonify(logged_in=is_logged_in, role=user_role)

@app.route('/doctors-patients', methods=['GET'])
def get_doctors_patients():
    doctor_id = request.args.get('doctor_id')
    if doctor_id in doctors_patients:
        return jsonify(doctors_patients[doctor_id])
    else:
        return jsonify([]), 404

# http://localhost:5001/doctors-patients?doctor_id=104405107080836112407

@app.route('/patients-records', methods=['GET'])
def get_patient_records():
    # Get the list of patient names from the query parameter
    patient_names = request.args.getlist('patient_names[]')

    filtered_records = [record for record in patients_records if record['name'] in patient_names]
    return jsonify(filtered_records)

@app.route('/search-by-patient', methods=['GET'])
def search_by_patient():
    query_name = request.args.get('name')
    filtered_records = [record for record in patients_records if query_name.lower() in record['name'].lower()]
    return jsonify(filtered_records)

# Office Hour Setting API Endpoints
@app.route('/officehours', methods=['GET', 'POST'])
def office_hours():
    if request.method == 'POST':
        data = request.json
        doctor_name = data['doctor_name']
        slots = data['slots']
        doctors_office_hours[doctor_name] = slots
        return jsonify({"status": "success"}), 200

    # If no doctor_name is provided, return all office hours
    doctor_name = request.args.get('doctor_name')
    if doctor_name:
        if doctor_name in doctors_office_hours:
            return jsonify(doctors_office_hours[doctor_name]), 200
        else:
            return jsonify([]), 200
    else:
        # Return all doctors' office hours
        return jsonify(doctors_office_hours), 200

# Appointment Scheduling API Endpoints
@app.route('/appointments', methods=['GET'])
def get_appointments():
    print("/appointments endpoint accessed. Request to GET all appointments.")
    print(session)
    '''
    if 'google_id' not in session:
        # 用户未登录，重定向到登录页面
        return redirect('/login')
    '''
    # user_role = session.get('role', 'not logged in')
    # return render_template('index.html', user_role=user_role)
    return jsonify(appointments)

@app.route('/appointments', methods=['POST'])
def add_appointments():

    '''
    if 'google_id' not in session:
        # 用户未登录，重定向到登录页面
        return redirect('/login')
    '''

    print("/appointments endpoint accessed. Request to POST a new appointments.")
    print(session)

    new_appointments = request.json
    new_id = max(appointment['id'] for appointment in appointments) + 1
    new_appointments['id'] = new_id
    appointments.append(new_appointments)

    # new_appointment_added = appointments[len(appointments)-1]
    # print("new appointment added", new_appointment_added)

    # API Gateway
    api_gateway_endpoint = "https://4t4vz1aovg.execute-api.us-east-1.amazonaws.com/default/test2"
    api_gateway_data = {
        "id": new_appointments['id'],
        "patientName": new_appointments['patientName'],
        "doctorName": new_appointments['doctorName'],
        "dateTime": new_appointments['dateTime']
    }

    response = requests.post(api_gateway_endpoint, json=api_gateway_data)
    print(response)
    return jsonify({"message": "Appointments added successfully"})

@app.route('/appointments/<int:appointment_id>', methods=['PUT'])
def update_appointment(appointment_id):
    '''
    if 'google_id' not in session:
        # 用户未登录，重定向到登录页面
        return redirect('/login')
    '''

    updated_appointment = request.json
    for appointment in appointments:
        if appointment["id"] == appointment_id:
            appointment.update(updated_appointment)
            if 'status' in appointment:
                appointment['status'] = appointment['status']
            return jsonify({"message": "Appointment updated successfully"})

    return jsonify({"error": "Appointment not found"}), 404

@app.route('/appointments/<int:appointment_id>', methods=['DELETE'])
def delete_appointment(appointment_id):

    '''
    if 'google_id' not in session:
        # 用户未登录，重定向到登录页面
        return redirect('/login')
    '''

    for appointment in appointments:
        if appointment["id"] == appointment_id:
            appointments.remove(appointment)
            return jsonify({"message": "Appointment deleted successfully"})

    return jsonify({"error": "Appointment not found"}), 404

@app.route('/appointments/<int:appointment_id>', methods=['GET'])
def get_appointment(appointment_id):

    '''
    if 'google_id' not in session:
        # 用户未登录，重定向到登录页面
        return redirect('/login')
    '''

    # Search for the appointment by ID
    appointment = next((appt for appt in appointments if appt['id'] == appointment_id), None)
    if appointment:
        return jsonify(appointment)
    else:
        return jsonify({"error": "Appointment not found"}), 404


if __name__ == '__main__':
    '''
    mongo_url = "mongodb+srv://lsy:lsy@cluster0.lwwlfix.mongodb.net/"
    client = pymongo.MongoClient(mongo_url)
    collection = client['6156']
    '''
    app.run(host='0.0.0.0', port=5001)


