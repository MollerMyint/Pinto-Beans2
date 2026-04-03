// App.js

import React, { useState, useEffect } from "react";
import "./App.css";
import QuestionBox from "./QuestionBox";


function App() {
    // Store the data we get back from the Flask API
    const [userData, setData] = useState({
        question: "",
    });
    const endpointStart = "http://localhost:3000/";

    // Run once when the component first loads
    useEffect(() => {
        fetch("/userData")
            .then((response) => response.json())
            .then((userData) => {
                setData({
                    question: userData.question,
                });
            })
            .catch((error) => {
                console.error("Error fetching data:", error);
            });
    }, []);
    
    const [question, setQuestion] = useState("");

    const submitQuestion = async () => {
    console.log("Button clicked");
    try {
        const response = await fetch("/askQ", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ question }),
        });

        const data = await response.json();
        console.log("Server said:", data);
    } catch (error) {
        console.error("Error:", error);
    }
};

    return (
        <div className="App">
            <header className="App-header">
                <h1>CPP AI Chatbot</h1>
                <p>Welcome to our chatbot</p>
                <button onClick={submitQuestion}>Submit</button>
                <QuestionBox/>
            </header>
            
        </div>

    );
}

export default App;