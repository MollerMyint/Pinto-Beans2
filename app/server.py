from flask import Flask, request, render_template
import agent
import mysql.connector
from dotenv import load_dotenv
import os
import hashlib

# Flask constructor 
app = Flask(__name__)

# Load the dotenv file 
load_dotenv()

mycursor = None 
pwd = None
s = None

def main():
    print("Starting Flask server...")

    try:
        mydb = mysql.connector.connect(
            host=os.getenv("DB_HOSTNAME"),
            user=os.getenv("DB_USERNAME"),
            password=os.getenv("DB_PASSWORD"),
            connection_timeout=5, 
            ssl_ca=os.getenv("DB_SSL"), 
            port=os.getenv("DB_PORT"),
            database=os.getenv("DB_DATABASE")
        )

        # Load salt and password for hashing 
        pwd = os.getenv("PWD")
        s = os.getenv("SALT")

        mycursor = mydb.cursor()
        mycursor.execute("SELECT * FROM users")
        result = mycursor.fetchall()

        print(result[0])

    except Exception as e:
        print("Database error:", e)

    app.run(debug=True, port=5001)


@app.route('/')
def home():
    return render_template("login.html")

# Tells the application which URL is associated with this function
@app.route('/ask', methods=['POST'])
def ask_question():
    question = request.form['corpus_question'] # get the data from the HTML form 
    
    # Use the function from agent.py to get the response
    answer = agent.search_corpus(question)
    return render_template("index.html", answer=answer)

@app.route('/login', methods=['POST'])
def login(): 
    username = request.form['username'] # get data from the HTML form 
    password = request.form['password'] # get the data from the HTML form 

    mycursor.execute()
 
@app.route('/signup', method=["POST"])
def signup():

    # Get data from the HTML form
    username = request.form['username']
    email = request.form['email']
    password = request.form['password']

if __name__ == "__main__":
    main()