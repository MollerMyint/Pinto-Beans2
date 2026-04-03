from flask import Flask, request, jsonify
import datetime
import agent

app = Flask(__name__)

x = datetime.datetime.now()

@app.route('/')
def home():
    return "It works!"

@app.route('/data')
def get_time(): 
    # returning an API for showing in reactjs 
    return {
        'Name': "geek", 
        "Age": "22", 
        "Date": x.strftime("%Y-%m-%d %H:%M:%S"), 
        "programming": "python"
    }


@app.route('/askQ', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get("question")

    print(f"I have received your question: {question}")

    return jsonify({"message": "Question received", "question": question})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)