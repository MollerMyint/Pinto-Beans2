from flask import Flask, request, render_template, jsonify, session, redirect
from langchain_core.messages import HumanMessage, AIMessage
from agent.agent import create_agent
import mysql.connector
from dotenv import load_dotenv
import os
import hashlib
import re

# Flask constructor 
app = Flask(__name__)

app.secret_key = "pinto-beans"

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

@app.route('/chatUI')
def chatUI():
    return render_template("index.html")

@app.route('/discord/ask', methods=['POST'])
def ask_discord_agent():

    data = request.get_json() # get the data from the HTML form 
    question = data.get("question")

    agent_executor = create_agent()  # Create a new agent to answer the question 

    response = agent_executor.invoke({"input": question, "chat_history": []})  # Use the function from agent.py to get the response
    full_answer = response['output']
    print(full_answer)
    parsed_answer = parse_agent_reply(full_answer) 
    answer = parsed_answer["answer"]

    return jsonify({"answer": answer})

# Tells the application which URL is associated with this function
@app.route('/ask', methods=['POST'])
def ask_question():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    
    data = request.get_json() # get the data from the HTML form 
    question = data.get("question")
    chat_id = data.get("chat_id")

    if not chat_id:
        return jsonify({"error": "No chat ID provided."}), 400
    
    mycursor.execute("SELECT question, answer FROM messages WHERE chat_id = %s ORDER BY message_id ASC", (chat_id,))
    rows = mycursor.fetchall()
    # print("rows: ", rows)

    chat_history = [] # rebuild chat history for agent
    for ques, answ in rows:
        chat_history.append(HumanMessage(content=ques))
        chat_history.append(AIMessage(content=answ))

    agent_executor = create_agent()  # Create a new agent to answer the question 

    response = agent_executor.invoke({"input": question, "chat_history": chat_history})  # Use the function from agent.py to get the response
    full_answer = response['output']
    print(full_answer)
    parsed_answer = parse_agent_reply(full_answer) 
    answer = parsed_answer["answer"]
    confidence = parsed_answer["confidence"] or 0

    # save the new Q&A to the messages table
    mycursor.execute("INSERT INTO messages (chat_id, question, answer, confidence) VALUES (%s, %s, %s, %s)",(chat_id, question, answer, confidence))
    mydb.commit()

    return jsonify({"answer": answer, "chat_id": chat_id})

@app.route('/new/chat', methods=['POST'])
def new_chat():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    
    data = request.get_json()
    question = data.get("question")

    try:
        agent_executor = create_agent()

        response = agent_executor.invoke({"input": question, "chat_history": []})  # Use the function from agent.py to get the response
        full_answer = response['output']
        print(full_answer)
        parsed_answer = parse_agent_reply(full_answer) # parse text to split title, answer, confidence

        title = parsed_answer["title"] or "New Chat"
        answer = parsed_answer["answer"]
        confidence = parsed_answer["confidence"] or 0

        # insert into tables and extract chat_id
        mycursor.execute("INSERT INTO chats (user_id, title) VALUES (%s, %s)", (user_id, title))
        chat_id = mycursor.lastrowid  # get the ID of the newly created chat
        mycursor.execute("INSERT INTO messages (chat_id, question, answer, confidence) VALUES (%s, %s, %s, %s)",(chat_id, question, answer, confidence))
        mydb.commit()  # Save the new row in the database

        return jsonify({"chat_id": chat_id, "title": title, "answer": answer, "confidence": confidence})

    except Exception as e:
        print("new_chat error:", e)
        return jsonify({"error": "Something went wrong creating the chat"}), 500

@app.route('/delete/chat/<int:chat_id>', methods=['DELETE'])
def delete_chat(chat_id):
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    
    # make sure the chat exisits
    mycursor.execute("SELECT chat_id FROM chats WHERE chat_id = %s AND user_id = %s", (chat_id, user_id))
    chat = mycursor.fetchone()
    if not chat:
        return jsonify({"error:", "Chat not found"}), 404

    # delete row(s) from chats and messages tables
    mycursor.execute("DELETE FROM messages WHERE chat_id = %s", (chat_id,))
    mycursor.execute("DELETE FROM chats WHERE chat_id = %s AND user_id = %s", (chat_id, user_id))
    mydb.commit()    
    return jsonify({"message": "Chat deleted successfully"})

@app.route('/change/title/<int:chat_id>', methods=["PUT"])
def change_title(chat_id):
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    
    data = request.get_json()
    new_title = data.get("title")

    mycursor.execute("SELECT chat_id FROM chats WHERE chat_id = %s AND user_id = %s", (chat_id, user_id))
    chat = mycursor.fetchone()
    if not chat:
        return jsonify({"error:", "Chat not found"}), 404
    
    mycursor.execute("UPDATE chats SET title = %s WHERE chat_id = %s AND user_id = %s", (new_title, chat_id, user_id))
    mydb.commit()

    return jsonify({"chat_id": chat_id, "title": new_title})

# return all of the chats that belong to a user
@app.route('/user/chats', methods=['GET'])
def get_chats():
    user_id = session.get("user_id") # get user_id from session
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    
    mycursor.execute("SELECT chat_id, title, created_at FROM chats WHERE user_id = %s ORDER BY created_at DESC", (user_id,))
    rows = mycursor.fetchall()

    chats = []
    for row in rows:
        chats.append({"chat_id": row[0], "title": row[1], "created_at": str(row[2])})
    
    return jsonify({"chats": chats})

# returns all of the messages that belong to a chat
@app.route('/chats/<int:chat_id>/messages', methods=['GET'])
def get_messages(chat_id):
    mycursor.execute("SELECT message_id, question, answer FROM messages WHERE chat_id = %s ORDER BY message_id ASC", (chat_id,))
    rows = mycursor.fetchall()
    # print(rows)

    messages = []
    for row in rows:
        messages.append({"message_id": row[0], "question": row[1], "answer": row[2]})
    
    return jsonify({"chat_id": chat_id, "messages": messages})

@app.route('/login', methods=['GET','POST'])
def login():

    # make sure the route is a proper fetch or post
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # get stored password
        mycursor.execute("SELECT user_id, password FROM users WHERE username = %s",(username,)) # check to see whether the passwords match
        result = mycursor.fetchone() # get the first value from the response
        
        if result:
            user_id, stored_password = result
            if hashPassword(password) == stored_password:
                session["user_id"] = user_id
                return redirect("/chatUI") # go to the home page if successfully logged in 
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

            user_id = mycursor.lastrowid
            session["user_id"] = user_id

            # If successful login, go to the homepage
            return redirect("/chatUI")
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

def parse_agent_reply(reply):
    reply = (reply or "").strip()

    title = None
    confidence = None
    answer = reply

    # extract title from the reply and strip it from the answer
    title_match = re.match(r'^\s*\*\*(.*?)\*\*\s*', answer, re.DOTALL)
    if title_match:
        title = title_match.group(1).strip()
        answer = answer[title_match.end():].strip()
    
    # extract confidence from the reply anf strip it from the aswer
    confidence_match = re.search(r'\n?\s*Confidence score:\s*(\d{1,3})%\s*$', answer, re.IGNORECASE)
    if confidence_match:
        confidence = int(confidence_match.group(1))
        answer = answer[:confidence_match.start()].strip()
    
    return {"title": title, "answer": answer, "confidence": confidence}

def main():
    print("Starting Flask server...")

    app.run(debug=True, port=5001)

# Run from native file
if __name__ == "__main__":
    main()