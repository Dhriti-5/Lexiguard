import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { AuthProvider } from "./contexts/AuthContext";
import Layout from "./components/Layout";
import ProtectedRoute from "./components/ProtectedRoute";
import Home from "./pages/Home";
import Upload from "./pages/Upload";
import Results from "./pages/Results";
import JobResults from "./pages/JobResults"; // ✅ ADD THIS IMPORT
import ChatWithDocument from "./pages/ChatWithDocument"; // ✅ ADD CHAT PAGE IMPORT
import Dashboard from "./pages/Dashboard";
import Glossary from "./pages/Glossary";
import FAQ from "./pages/FAQ";
import About from "./pages/About";
import Contact from "./pages/Contact";
import Login from "./pages/Login";
import LawyerDirectory from "./pages/LawyerDirectory";
function App() {
  return (
    <Router>
      <AuthProvider>
        <Layout>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/login" element={<Login />} />
            <Route path="/about" element={<About />} />
            <Route path="/glossary" element={<Glossary />} />
            <Route path="/lawyers" element={<LawyerDirectory />} />
            <Route path="/faq" element={<FAQ />} />
            <Route path="/contact" element={<Contact />} />
            
            {/* Protected Routes - Require Authentication */}
            <Route
              path="/upload"
              element={
                <ProtectedRoute>
                  <Upload />
                </ProtectedRoute>
              }
            />
            
            {/* ✅ ADD THIS NEW ROUTE - Job Status Tracking for Async Uploads */}
            <Route
              path="/results/job/:jobId"
              element={
                <ProtectedRoute>
                  <JobResults />
                </ProtectedRoute>
              }
            />
            
            {/* Existing Results Routes */}
            <Route
              path="/results"
              element={
                <ProtectedRoute>
                  <Results />
                </ProtectedRoute>
              }
            />
            <Route
              path="/results/:analysisId"
              element={
                <ProtectedRoute>
                  <Results />
                </ProtectedRoute>
              }
            />
            
            {/* ✅ ADD CHAT WITH DOCUMENT ROUTE */}
            <Route
              path="/chat/:analysisId"
              element={
                <ProtectedRoute>
                  <ChatWithDocument />
                </ProtectedRoute>
              }
            />
            
            <Route
              path="/dashboard"
              element={
                <ProtectedRoute>
                  <Dashboard />
                </ProtectedRoute>
              }
            />
          </Routes>
        </Layout>
      </AuthProvider>
    </Router>
  );
}

export default App;