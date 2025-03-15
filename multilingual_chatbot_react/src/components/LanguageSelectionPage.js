import React from "react";
import "../App.css";

const languages = ["English", "Hindi", "Tamil", "Telugu", "Kannada"];

const LanguageSelectionPage = ({ setSelectedLanguage }) => {
  return (
    <div className="language-container">
      <h2>Select Your Language</h2>
      {languages.map((lang) => (
        <button key={lang} onClick={() => setSelectedLanguage(lang)}>
          {lang}
        </button>
      ))}
    </div>
  );
};

export default LanguageSelectionPage;