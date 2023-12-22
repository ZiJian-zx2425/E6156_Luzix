import pathlib

# import pymongo
# from database import *
from openai import OpenAI
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

from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)

load_dotenv()

my_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=my_api_key)
# openai.api_key = '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'

# SSO logic
# print(f"The name of the Flask application is: {app_name}")
app.secret_key = "CodeSpecialist.com"
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

GOOGLE_CLIENT_ID = "665838449619-o9ijkr28tubs3uhpjdk2aornhttpe5la.apps.googleusercontent.com"
client_secrets_file = os.path.join(pathlib.Path(__file__).parent, "client_secret.json")

users_db = {
    '110788464327696265201': 'patient',
    '104405107080836112407': 'doctor',
    '117740487543455173970': 'volunteer',
}

doctors = [
    {'id': 1, 'name': 'Dr. Smith'},
    {'id': 2, 'name': 'Dr. Jones'}
]

contact_messages = []



flow = Flow.from_client_secrets_file(
    client_secrets_file=client_secrets_file,
    scopes=["https://www.googleapis.com/auth/userinfo.profile", "https://www.googleapis.com/auth/userinfo.email", "openid"],
    #TODO: change after cloud deployment
    redirect_uri="http://127.0.0.1:5002/callback"
)

def login_is_required(function):
    def wrapper(*args, **kwargs):
        if "google_id" not in session:
            return abort(401)  # Authorization required
        else:
            return function()
    return wrapper



@app.route('/')
def home():
    return "Chat Microservice is running"

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

    print("before callback redirect return statement")
    # return redirect('/ask')
    return redirect(f"http://localhost:4200/auth-success?token={token}")

    # OLD S3 BUCKET: appointment-scheduler-angular
    # return redirect(f"http://scheduler-frontend.s3-website-us-east-1.amazonaws.com/auth-success?token={token}")
    # return redirect(f"http://scheduler-frontend.s3-website-us-east-1.amazonaws.com")
    # NEW S3 BUCKET: appointment-scheduler-angular
    # return redirect(f"http://appointment-scheduler-angular.s3-website-us-east-1.amazonaws.com")

    # TODO: change after deployment
    # return redirect(f"http://appointment-scheduler-angular.s3-website-us-east-1.amazonaws.com/auth-success?token={token}")

def create_token(user_id, user_name, user_role):
    payload = {
        'exp': datetime.datetime.utcnow() + datetime.timedelta(days=1),  # Token expiration time
        'iat': datetime.datetime.utcnow(),  # Issued at time
        'sub': user_id,
        'name': user_name,
        'role': user_role,
        # 'iss': 'flask_backend',
        # 'aud': 'apiGateway'
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

@app.route('/api/is_logged_in')
def is_logged_in():
    is_logged_in = 'google_id' in session
    user_role = session.get('role', 'not logged in') if is_logged_in else 'not logged in'
    return jsonify(logged_in=is_logged_in, role=user_role)




@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        # Get the user question from the request JSON
        data = request.get_json()
        user_question = data.get('question', 'Default question if none provided')
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "You are a health assitant."},
                {"role": "user", "content": user_question }
            ]
        )
        # answer = response.choices[0].text.strip()
        # return jsonify({"answer": answer}), 200
        message_text = completion.choices[0].message.content
        return jsonify({"answer": message_text}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/contact-doctor', methods=['POST'])
def post_message():
    data = request.json
    # doctor = next((doctor for doctor in doctors if doctor['id'] == data['doctor_id']), None)
    # doctor_name = doctor['name'] if doctor else 'Unknown Doctor'
    new_message = {
        'id': len(contact_messages) + 1,
        'google_id': data['google_id'],
        'doctor_id': data['doctor_id'],
         # 'doctor_name': data['doctor_name'],
         # 'doctor_name': doctor_name,
        'message': data['message']
    }
    contact_messages.append(new_message)
    return jsonify(new_message), 201

@app.route('/contact-doctor/<int:message_id>', methods=['DELETE'])
def delete_message(message_id):
    global contact_messages
    contact_messages = [msg for msg in contact_messages if msg['id'] != message_id]
    return jsonify({"message": "Message deleted successfully"}), 200

@app.route('/contact-doctor/<int:message_id>', methods=['PUT'])
def update_message(message_id):
    print("update_message being hit")
    print(message_id)
    data = request.json
    for msg in contact_messages:
        if msg['id'] == message_id:
            msg['message'] = data['message']
            print("update message success")
            return jsonify(msg), 200
    return jsonify({"error": "Message not found"}), 404


@app.route('/contact-doctor/<google_id>', methods=['GET'])
def get_messages(google_id):
    print("getting messages for google:id: ", google_id)
    user_messages = [msg for msg in contact_messages if msg['google_id'] == google_id]
    return jsonify(user_messages), 200



if __name__ == '__main__':
    # app.run(debug=True)
    app.run(debug=True, host='0.0.0.0', port=5002)