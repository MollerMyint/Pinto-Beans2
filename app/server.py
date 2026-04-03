from flask import Flask, request, render_template
import agent

# Flask constructor 
app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

# Tells the application which URL is associated with this function
@app.route('/ask', methods=['POST'])
def ask_question():
    question = request.form['corpus_question'] # get the data from the HTML form 
    
    # Use the function from agent.py to get the response
    answer = agent.search_corpus(question)
    return render_template("index.html", answer=answer)

if __name__ == '__main__':
    app.run(debug=True, port=5001)