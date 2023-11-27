import requests
from flask import Flask, jsonify, request, abort
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

QUIVR_API_URL = 'https://api.quivr.app'  # Replace with the actual Quivr API endpoint
QUIVR_API_KEY = 'dadc858b79288bb7f23d004f45f86507'  # Replace with your Quivr API key

# https://www.quivr.app/user

@app.route('/create-brain', methods=['POST'])
def create_brain():
    headers = {
        'Authorization': f'Bearer {QUIVR_API_KEY}',
        'Content-Type': 'application/json'
    }
    # Construct the data payload according to Quivr's API expectations
    data = {
        'name': 'My Custom Brain',  # or use question if appropriate
        'description': 'A brain for answering health-related questions',
        'status': 'private',
        'model': 'gpt-3.5-turbo',  # Specify the model as per Quivr's documentation
    }
    response = requests.post(f'{QUIVR_API_URL}/brains/', headers=headers, json=data)
    # Handle the response from Quivr
    if response.status_code != 200:
        abort(response.status_code, response.text)
    brain_data = response.json()
    return jsonify(brain_data)


@app.route('/ask/<brain_id>', methods=['POST'])
def ask_question(brain_id):
    if not request.is_json:
        return jsonify({'error': 'Missing JSON in request'}), 400

    question = request.json.get('question', '')
    headers = {
        'Authorization': f'Bearer {QUIVR_API_KEY}',
        'Content-Type': 'application/json'
    }
    data = {
        'prompt': question,
    }

    try:
        response = requests.post(f'{QUIVR_API_URL}/brains/{brain_id}/query', headers=headers, json=data)
        response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code
    except requests.exceptions.HTTPError as http_err:
        # Log the error or send it back to the frontend
        return jsonify({'error': f'HTTP error occurred: {http_err}'}), response.status_code
    except Exception as err:
        # Handle other possible errors
        return jsonify({'error': f'Other error occurred: {err}'}), 500

    # If successful, return the JSON response from Quivr API
    return jsonify(response.json())





if __name__ == '__main__':
    app.run(debug=True)