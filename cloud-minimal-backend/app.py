from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)  # allows your Flask app to accept requests from any domain

# Access the name of the Flask application
app_name = app.name
# Print the name of the Flask application
print(f"The name of the Flask application is: {app_name}")


appointments = [
    {
        "id": 1,
        "patientName": "John Doe",
        "doctorName": "Dr. Smith",
        "dateTime": "2023-12-01T10:00"  # ISO 8601 date format
    },
    {
        "id": 2,
        "patientName": "Jane Smith",
        "doctorName": "Dr. Jones",
        "dateTime": "2023-12-02T15:00"  # ISO 8601 date format
    },
]

# Configure the database connection (replace with your RDS credentials)
# app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://shuuting:tripleshot123321@database-1.cyacyckrig0k.us-east-1.rds.amazonaws.com/dbname'
# db = SQLAlchemy(app)

@app.route('/')
def hello_world():
    return jsonify(message='Hello from your microservice!')

@app.route('/appointments', methods=['GET'])
def get_appointments():
    return jsonify(appointments)

@app.route('/appointments', methods=['POST'])
def add_appointments():
    new_appointments = request.json
    new_id = max(appointment['id'] for appointment in appointments) + 1  # Create a new unique ID
    new_appointments['id'] = new_id
    appointments.append(new_appointments)
    # return jsonify(new_appointment), 201
    return jsonify({"message": "Appointments added successfully"})

@app.route('/appointments/<int:appointment_id>', methods=['PUT'])
def update_appointment(appointment_id):
    updated_appointment = request.json
    for appointment in appointments:
        if appointment["id"] == appointment_id:
            appointment.update(updated_appointment)
            return jsonify({"message": "Appointment updated successfully"})
    return jsonify({"error": "Appointment not found"}), 404

@app.route('/appointments/<int:appointment_id>', methods=['DELETE'])
def delete_appointment(appointment_id):
    for appointment in appointments:
        if appointment["id"] == appointment_id:
            appointments.remove(appointment)
            return jsonify({"message": "Appointment deleted successfully"})
    return jsonify({"error": "Appointment not found"}), 404

@app.route('/appointments/<int:appointment_id>', methods=['GET'])
def get_appointment(appointment_id):
    # Search for the appointment by ID
    appointment = next((appt for appt in appointments if appt['id'] == appointment_id), None)
    if appointment:
        return jsonify(appointment)
    else:
        return jsonify({"error": "Appointment not found"}), 404


if __name__ == '__main__':
    # app.run()
    app.run(host='0.0.0.0', port=5000)
