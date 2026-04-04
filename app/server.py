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
    question = request.form['corpus_question'] # get the data from the HTML form 
    
    # Use the function from agent.py to get the response
    answer = agent.search_corpus(question)
    return render_template("index.html", answer=answer)

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
            return render_template("signup.html", error="Soemthing went wrong, please try again.")
    return render_template("signup.html")

# Helper functions 
def hashPassword(plainText):
    pwd_salt = plainText+s
    hashed = hashlib.sha256(pwd_salt.encode()).hexdigest()
    return hashed

def main():
    print("Starting Flask server...")

    app.run(debug=True, port=5001)



# Run from native file
if __name__ == "__main__":
    main()