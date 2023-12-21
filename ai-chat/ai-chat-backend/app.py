import requests
from flask import Flask, jsonify, request, abort
from flask_cors import CORS
# import openai from OpenAI
from openai import OpenAI

app = Flask(__name__)
CORS(app)

# client = OpenAI(api_key='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')

client = OpenAI(api_key='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')
# openai.api_key = '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'

@app.route('/')
def home():
    return "Microservice is running"


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
                 "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
                {"role": "user", "content": user_question }
            ]
        )
        # answer = response.choices[0].text.strip()
        # return jsonify({"answer": answer}), 200
        message_text = completion.choices[0].message.content
        return jsonify({"answer": message_text}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # app.run(debug=True)
    app.run(debug=True, host='0.0.0.0', port=5002)