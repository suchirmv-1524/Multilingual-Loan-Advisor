import React from 'react';
import { BrowserRouter as Router, Route, Switch } from 'react-router-dom';
import AuthPage from './components/AuthPage';
import LanguageSelectionPage from './components/LanguageSelectionPage';
import ChatbotPage from './components/ChatbotPage';

const App = () => {
  return (
    <Router>
      <Switch>
        <Route path="/login" component={AuthPage} />
        <Route path="/language" component={LanguageSelectionPage} />
        <Route path="/chatbot" component={ChatbotPage} />
        <Route path="/" component={AuthPage} />
      </Switch>
    </Router>
  );
};

export default App;
