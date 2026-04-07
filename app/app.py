from flask import Flask, request, render_template, jsonify
from agent.agent import create_agent
import mysql.connector
from dotenv import load_dotenv
import os
import hashlib
import re

# Flask constructor 
app = Flask(__name__)

# Load the dotenv file 
load_dotenv()

# Set up the database connection 
try:
    
    # Grab environment variables from the dotenv file
    mydb = mysql.connector.connect(
        host=os.getenv("DB_HOSTNAME"),
        user=os.getenv("DB_USERNAME"),
        password=os.getenv("DB_PASSWORD"),
        connection_timeout=5, 
        ssl_ca=os.getenv("DB_SSL"), 
        port=os.getenv("DB_PORT"),
        database=os.getenv("DB_DATABASE")
    )

    # Create the cursor to iterate through the database
    mycursor = mydb.cursor()

    # Testing if the database works 
    mycursor.execute("SELECT * FROM users")
    result = mycursor.fetchall()

except Exception as e:
    print("Database error:", e)

# Load salt and password for hashing 
s = os.getenv("SALT")

# App routes with functions 
@app.route('/')
def home():
    return render_template("login.html")

# Tells the application which URL is associated with this function
@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json() # get the data from the HTML form 
    question = data.get("corpus_question", "")
    chat_history = []

    # Create a new agent to answer the question 
    agent_executor = create_agent()

    # Use the function from agent.py to get the response
    response = agent_executor.invoke({
                "input": question,
                "chat_history": chat_history
    })
    return jsonify({"answer":response['output']})

@app.route('/login', methods=['GET','POST'])
def login():

    # make sure the route is a proper fetch or post
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # get stored password
        mycursor.execute("SELECT password FROM users WHERE username = %s",(username,)) # check to see whether the passwords match
        result = mycursor.fetchone() # get the first value from the response
        
        if result:
            stored_password = result[0]
            if hashPassword(password) == stored_password:
                return render_template("index.html") # go to the home page if successfully logged in 
        return render_template("login.html", error="Invalid username or password") # if login wasn't successful, stay on the log in page
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        try: 
            username = request.form['username']
            email = request.form['email']
            password = request.form['password']

            mycursor.execute("SELECT username, emailaddress FROM users WHERE username = %s OR emailaddress = %s", (username, email))
            existing = mycursor.fetchall()

            username_error = None
            email_error = None

            # check for dupilcate username or email
            for row in existing:
                if row[0] == username:
                    username_error = "That username is already taken."
                if row[1] == email:
                    email_error = "An account with that email already exists."

            # validate email format
            if not is_valid_email(email):
                email_error = "Please enter a valid email address."

            if username_error or email_error:
                # pass back username and email so fields dont clear when error is present
                return render_template("signup.html",username_error=username_error, email_error=email_error, username=username, email=email)
            
            # hash the password for added security
            password = hashPassword(password)

            # Insert the new account information into the database
            mycursor.execute("INSERT INTO users (username, emailaddress, password) VALUES (%s, %s, %s)", (username, email, password))
            # Save the new row in the database
            mydb.commit()

            # If successful login, go to the homepage
            return render_template("index.html")
        except Exception as e: 
            # If there's an issue, stay on the signup page until it works
            print("Signup error:", e)
            return render_template("signup.html", error="Something went wrong, please try again.")
    return render_template("signup.html")

# Helper functions 
def hashPassword(plainText):
    pwd_salt = plainText+s
    hashed = hashlib.sha256(pwd_salt.encode()).hexdigest()
    return hashed

def is_valid_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def main():
    print("Starting Flask server...")

    app.run(debug=True, port=5001)



# Run from native file
if __name__ == "__main__":
    main()