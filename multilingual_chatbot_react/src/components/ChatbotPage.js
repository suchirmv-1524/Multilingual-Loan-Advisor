import React, { useState } from "react";
import "../App.css";

const ChatbotPage = ({ language }) => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");

  const sendMessage = () => {
    if (input.trim()) {
      setMessages([...messages, { text: input, sender: "user" }]);
      setInput("");
      // Simulate chatbot response
      setTimeout(() => {
        setMessages((prevMessages) => [...prevMessages, { text: "This is a response", sender: "bot" }]);
      }, 1000);
    }
  };

  return (
    <div className="chatbot-container">
      <h2>Chatbot - {language}</h2>
      <div className="chatbox">
        {messages.map((msg, index) => (
          <p key={index} className={msg.sender}>{msg.text}</p>
        ))}
      </div>
      <input type="text" value={input} onChange={(e) => setInput(e.target.value)} placeholder="Type a message..." />
      <button onClick={sendMessage}>Send</button>
    </div>
  );
};

export default ChatbotPage;
