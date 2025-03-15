// src/App.js
import React, { useState } from "react";
import { BrowserRouter as Router, Route, Routes, Navigate } from "react-router-dom";
import AuthPage from "./components/AuthPage";
import LanguageSelectionPage from "./components/LanguageSelectionPage";
import ChatbotPage from "./components/ChatbotPage";
import Header from "./components/Header"; // Import Header component
import "./App.css";

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [selectedLanguage, setSelectedLanguage] = useState(null);

  return (
    <Router>
      <Header /> {/* Add Header component at the top */}
      <Routes>
        <Route
          path="/"
          element={
            isAuthenticated ? (
              selectedLanguage ? (
                <Navigate to="/chatbot" />
              ) : (
                <Navigate to="/language" />
              )
            ) : (
              <AuthPage setIsAuthenticated={setIsAuthenticated} />
            )
          }
        />
        <Route
          path="/language"
          element={<LanguageSelectionPage setSelectedLanguage={setSelectedLanguage} />}
        />
        <Route
          path="/chatbot"
          element={selectedLanguage ? <ChatbotPage language={selectedLanguage} /> : <Navigate to="/language" />}
        />
      </Routes>
    </Router>
  );
}

export default App;
