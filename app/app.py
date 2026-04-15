from flask import Flask, request, render_template, jsonify, session, redirect
from langchain_core.messages import HumanMessage, AIMessage
from agent.agent import create_agent, get_sbert_model, load_sbert_index
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

@app.route('/account')
def account():
    user_id = session.get("user_id")
    if not user_id:
        return redirect("/")
    
    mycursor.execute("SELECT username, emailaddress FROM users WHERE user_id = %s", (user_id,))
    result = mycursor.fetchone()

    return render_template("account.html", username=result[0], email=result[1])

@app.route('/chatUI')
def chatUI():
    user_id = session.get("user_id")
    mycursor.execute("SELECT username FROM users WHERE user_id = %s", (user_id,))
    result = mycursor.fetchone()
    username = result[0] if result else "User"
    return render_template("index.html", username=username)

@app.route('/discord/ask', methods=['POST'])
def ask_discord_agent():

    data = request.get_json() # get the data from the HTML form 
    question = data.get("question")

    agent_executor = create_agent()  # Create a new agent to answer the question 

    response = agent_executor.invoke({"input": question, "chat_history": []})  # Use the function from agent.py to get the response
    answer = response['output']
    print(answer)

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
        return jsonify({"error": "No chat ID provided"}), 400
    
    mycursor.execute("SELECT question, answer FROM messages WHERE chat_id = %s ORDER BY message_id ASC", (chat_id,))
    rows = mycursor.fetchall()
    # print("rows: ", rows)

    chat_history = [] # rebuild chat history for agent
    for ques, answ in rows:
        chat_history.append(HumanMessage(content=ques))
        chat_history.append(AIMessage(content=answ))

    agent_executor = create_agent()  # Create a new agent to answer the question 

    response = agent_executor.invoke({"input": question, "chat_history": chat_history})  # Use the function from agent.py to get the response
    answer = response['output']
    print(answer)

    # save the new Q&A to the messages table
    mycursor.execute("INSERT INTO messages (chat_id, question, answer) VALUES (%s, %s, %s)",(chat_id, question, answer,))
    message_id = mycursor.lastrowid #row ID
    # update last activity time
    mycursor.execute("UPDATE chats SET created_at = CURRENT_TIMESTAMP WHERE chat_id = %s",(chat_id,))
    mydb.commit()

    return jsonify({"answer": answer, "chat_id": chat_id, "message_id": message_id})

@app.route('/new/chat', methods=['POST'])
def new_chat():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    
    data = request.get_json()
    question = data.get("question")

    try:
        agent_executor = create_agent(include_title_tool=True)

        response = agent_executor.invoke({"input": question, "chat_history": []})  # Use the function from agent.py to get the response
        full_answer = response['output']
        print(full_answer)

        title = None
        answer = full_answer

        # extract title from the reply and strip it from the answer
        title_match = re.match(r'^\s*\*\*(.*?)\*\*\s*', answer, re.DOTALL)
        if title_match:
            title = title_match.group(1).strip().capitalize()
            answer = answer[title_match.end():].strip()

        # insert into tables and extract chat_id
        mycursor.execute("INSERT INTO chats (user_id, title) VALUES (%s, %s)", (user_id, title))
        chat_id = mycursor.lastrowid  # get the ID of the newly created chat
        mycursor.execute("INSERT INTO messages (chat_id, question, answer) VALUES (%s, %s, %s)",(chat_id, question, answer,))
        message_id = mycursor.lastrowid #row ID
        mydb.commit()  # Save the new row in the database

        return jsonify({"chat_id": chat_id, "title": title, "answer": answer, "message_id": message_id})

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

@app.route('/change/message/<int:message_id>', methods=["PUT"])
def change_chat(message_id):
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    
    data = request.get_json()
    new_question = data.get("question")
    new_prompt = new_question + " Please expand on this more and change up the wording."
    print(new_prompt)

    agent_executor = create_agent()
    response = agent_executor.invoke({"input": new_prompt, "chat_history": []})
    new_answer = response['output']

    mycursor.execute("SELECT chat_id FROM messages WHERE message_id = %s",(message_id,))
    row = mycursor.fetchone()
    if not row:
        return jsonify({"error": "Message not found"}), 404
    chat_id = row[0]

    mycursor.execute("UPDATE messages SET question = %s, answer = %s WHERE message_id = %s", (new_question, new_answer, message_id))
    mycursor.execute("UPDATE chats SET created_at = CURRENT_TIMESTAMP WHERE chat_id = %s",(chat_id,))
    mydb.commit()

    return jsonify({"message_id": message_id, "answer": new_answer, "question": new_question})

@app.route('/change/username', methods=['PUT'])
def change_username():
    user_id = session.get("user_id") # get user_id from session
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    
    data = request.get_json()
    new_username = data.get("username", "").strip()

    if not new_username:
        return jsonify({"error": "Username cannot be empty"}), 400

    mycursor.execute("SELECT user_id FROM users WHERE username = %s", (new_username,))
    if mycursor.fetchone():
        return jsonify({"error": "That username is already taken"}), 409

    mycursor.execute("UPDATE users SET username = %s WHERE user_id = %s", (new_username, user_id))
    mydb.commit()
    return jsonify({"message": "Username updated successfully", "username": new_username})

@app.route('/change/email', methods=['PUT'])
def change_email():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    data = request.get_json()
    new_email = data.get("email", "").strip()

    email_error = validate_email(new_email)
    if email_error:
        return jsonify({"error": email_error}), 400

    mycursor.execute("SELECT user_id FROM users WHERE emailaddress = %s", (new_email,))
    if mycursor.fetchone():
        return jsonify({"error": "An account with that email already exists"}), 409

    mycursor.execute("UPDATE users SET emailaddress = %s WHERE user_id = %s", (new_email, user_id))
    mydb.commit()
    return jsonify({"message": "Email updated successfully", "email": new_email})

@app.route('/change/password', methods=['PUT'])
def change_password():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    data = request.get_json()
    old_password = data.get("old_password", "")
    new_password = data.get("new_password", "")
    confirm_password = data.get("confirm_password", "")

    # get stored password to validate
    mycursor.execute("SELECT password FROM users WHERE user_id = %s", (user_id,))
    result = mycursor.fetchone()
    if not result:
        return jsonify({"error": "User not found"}), 404

    password_error = validate_password(old_password, new_password, confirm_password, result[0])
    if password_error:
        return jsonify({"error": password_error}), 400  

    mycursor.execute("UPDATE users SET password = %s WHERE user_id = %s", (hashPassword(new_password), user_id))
    mydb.commit()
    return jsonify({"message": "Password updated successfully"}) 

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
                    username_error = "That username is already taken"
                if row[1] == email:
                    email_error = "An account with that email already exists"

            # validate email format
            if not is_valid_email(email):
                email_error = "Please enter a valid email address"

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
            return render_template("signup.html", error="Something went wrong, please try again")
    return render_template("signup.html")

# Route to logout
@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

@app.route('/delete/account', methods=['DELETE'])
def delete_account():
    user_id = session.get("user_id") # get user_id from session
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    
    # delete messages for all of this user's chats
    mycursor.execute("SELECT chat_id FROM chats WHERE user_id = %s", (user_id,))
    chats = mycursor.fetchall()

    for chat in chats:
        mycursor.execute("DELETE FROM messages WHERE chat_id = %s", (chat[0],))

    # delete chats and user
    mycursor.execute("DELETE FROM chats WHERE user_id = %s", (user_id,))
    mycursor.execute("DELETE FROM users WHERE user_id = %s", (user_id,))
    mydb.commit()

    session.clear()  # log the user out after deletion
    return jsonify({"message": "Account deleted successfully"})

# Helper functions 
def hashPassword(plainText):
    pwd_salt = plainText+s
    hashed = hashlib.sha256(pwd_salt.encode()).hexdigest()
    return hashed

def validate_password(old_password, new_password, confirm_password, stored_hash):
    if not old_password or not new_password or not confirm_password:
        return "All password fields are required"
    if hashPassword(old_password) != stored_hash:
        return "Old password is incorrect"
    if new_password != confirm_password:
        return "New passwords do not match"
    if old_password == new_password:
        return "New password must be different from old password"
    return None

def validate_email(email):
    if not email:
        return "Email cannot be empty"
    if not is_valid_email(email):
        return "Please enter a valid email address"
    return None

def is_valid_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def preload_sbert_resources():
    """
    Warm SBERT resources once at process startup so request-time latency is lower.
    """
    try:
        get_sbert_model()
        load_sbert_index()
        print("SBERT model and embedding index preloaded.")
    except Exception as e:
        # Keep server boot resilient; agent tools still return graceful fallback messages.
        print(f"SBERT preload skipped: {e}")

def main():
    print("Starting Flask server...")
    preload_sbert_resources()

    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))

# Run from native file
if __name__ == "__main__":
    main()