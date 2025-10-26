import React, { useState, useEffect } from "react";
import { useLocation, useNavigate, useParams } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Badge } from "../components/ui/badge";
import BackToTop from "../components/BackToTop";
import { getAnalysisById } from "../services/firestoreService";
import LanguageSelector from "../components/LanguageSelector";
import { getTranslation, requestTranslation } from "../services/firestoreService";
// REMOVE THIS LINE: import MarkdownRenderer from "../components/MarkdownRenderer";
import {
  AlertTriangle,
  CheckCircle,
  FileText,
  MessageSquare,
  ArrowLeft,
  Send,
  Mail,
  Scale,
  TrendingUp,
  Shield,
  Copy,
  Check,
  X,
  ChevronDown,
  ChevronUp,
  Loader2,
  Globe,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

const KEYWORDS_REGEX = /\b(agreement|contract|clause|liability|termination|penalty|indemnity|renewal)\b/gi;
const ACTIONABLE_REGEX = /\b(\d+\s*(days?|weeks?|months?|years?)|due by|deadline|penalty|fine|termination|payment of \$?\d+|effective date|must|shall|obligated|required)\b/gi;

// MarkdownRenderer Component
const MarkdownRenderer = ({ text }) => {
  if (!text || typeof text !== 'string') return null;

  const paragraphs = text.split('\n\n');

  const renderLine = (line, index) => {
   // Bold: **text**
line = line.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
// Italic: *text*
line = line.replace(/\*(.*?)\*/g, '<em>$1</em>');

    if (line.startsWith('### ')) {
      return <h3 key={index} className="text-lg font-bold text-white mt-4 mb-2" dangerouslySetInnerHTML={{ __html: line.replace('### ', '') }} />;
    }
    if (line.startsWith('## ')) {
      return <h2 key={index} className="text-xl font-bold text-white mt-4 mb-2" dangerouslySetInnerHTML={{ __html: line.replace('## ', '') }} />;
    }
    if (line.startsWith('# ')) {
      return <h1 key={index} className="text-2xl font-bold text-white mt-4 mb-2" dangerouslySetInnerHTML={{ __html: line.replace('# ', '') }} />;
    }
    
    if (line.trim().startsWith('- ') || line.trim().startsWith('* ')) {
      return (
        <li key={index} className="ml-4 text-gray-200" dangerouslySetInnerHTML={{ 
          __html: line.trim().replace(/^[-*]\s/, '') 
        }} />
      );
    }
    
    const numberedMatch = line.trim().match(/^\d+\.\s/);
    if (numberedMatch) {
      return (
        <li key={index} className="ml-4 text-gray-200" dangerouslySetInnerHTML={{ 
          __html: line.trim().replace(/^\d+\.\s/, '') 
        }} />
      );
    }
    
    if (line.trim()) {
      return <p key={index} className="text-gray-200 mb-2 leading-relaxed" dangerouslySetInnerHTML={{ __html: line }} />;
    }
    
    return null;
  };

  return (
    <div className="prose prose-invert max-w-none">
      {paragraphs.map((paragraph, pIndex) => {
        const lines = paragraph.split('\n');
        
        const isBulletList = lines.some(l => l.trim().startsWith('- ') || l.trim().startsWith('* '));
        const isNumberedList = lines.some(l => /^\d+\.\s/.test(l.trim()));
        
        if (isBulletList) {
          return (
            <ul key={pIndex} className="list-disc ml-6 mb-4 space-y-1">
              {lines.map((line, lIndex) => renderLine(line, `${pIndex}-${lIndex}`))}
            </ul>
          );
        }
        
        if (isNumberedList) {
          return (
            <ol key={pIndex} className="list-decimal ml-6 mb-4 space-y-1">
              {lines.map((line, lIndex) => renderLine(line, `${pIndex}-${lIndex}`))}
            </ol>
          );
        }
        
        return (
          <div key={pIndex} className="mb-4">
            {lines.map((line, lIndex) => renderLine(line, `${pIndex}-${lIndex}`))}
          </div>
        );
      })}
    </div>
  );
};

function highlightText(text) {
  if (!text || typeof text !== "string") return "";
  const parts = text.split(ACTIONABLE_REGEX);

  return parts.map((part, idx) => {
    if (!part) return null;

    if (ACTIONABLE_REGEX.test(part)) {
      return (
        <span key={idx} className="bg-red-500/50 text-white font-bold px-1 rounded">
          {part}
        </span>
      );
    } else {
      const subParts = part.split(KEYWORDS_REGEX);
      return subParts.map((subPart, subIdx) => {
        if (!subPart) return null;
        return KEYWORDS_REGEX.test(subPart) ? (
          <span key={`${idx}-${subIdx}`} className="bg-cyan-400/20 text-cyan-200 px-1 rounded">
            {subPart}
          </span>
        ) : (
          subPart
        );
      });
    }
  });
}
export default function Results() {
  const location = useLocation();
  const navigate = useNavigate();
  const { analysisId } = useParams();
  const { currentUser } = useAuth();
  
const [analysis, setAnalysis] = useState(() => {
  const state = location.state?.analysis;
  if (state) {
    console.log("📦 Initial analysis from location.state:", state);
  }
  return state || null;
});
  const [analysisType, setAnalysisType] = useState(location.state?.analysisType || null);
  const [loading, setLoading] = useState(false);

   const [chatMessage, setChatMessage] = useState("");
  const [chatHistory, setChatHistory] = useState([]);
  const [chatLoading, setChatLoading] = useState(false);
  const [selectedClauseForNegotiation, setSelectedClauseForNegotiation] = useState(null);
  const [negotiationEmail, setNegotiationEmail] = useState("");
  const [negotiationLoading, setNegotiationLoading] = useState(false);
  const [copiedEmail, setCopiedEmail] = useState(false);
  const [showDocumentEmail, setShowDocumentEmail] = useState(false);
  const [documentEmail, setDocumentEmail] = useState("");
  const [documentEmailLoading, setDocumentEmailLoading] = useState(false);
  const [copiedDocumentEmail, setCopiedDocumentEmail] = useState(false);
  const [userEmail, setUserEmail] = useState("");
  const [emailSending, setEmailSending] = useState(false);
  const [emailSent, setEmailSent] = useState(false);
  const [expandedClauses, setExpandedClauses] = useState({});
  
  // Translation states
  const [selectedLanguage, setSelectedLanguage] = useState('en');
  const [translatedContent, setTranslatedContent] = useState(null);
  const [translationLoading, setTranslationLoading] = useState(false);

  useEffect(() => {
    if (analysisId && currentUser && !analysis) {
      loadAnalysisFromFirestore();
    }
  }, [analysisId, currentUser]);
  useEffect(() => {
  // Reset to English whenever we load a new analysis
  setSelectedLanguage('en');
  setTranslatedContent(null);
  setTranslationLoading(false);
}, [analysisId]);

// ADD debug logging for displayedContent:
useEffect(() => {
  if (analysis) {
    const content = getDisplayedContent();
    console.log("🔍 Current displayed content:", {
      selectedLanguage,
      translationLoading,
      hasTranslatedContent: !!translatedContent,
      summary: content.summary?.substring(0, 100),
      risksCount: content.risks?.length,
      clausesCount: content.clauses?.length,
      analysisType
    });
  }
}, [analysis, selectedLanguage, translatedContent, translationLoading, analysisType]);

const loadAnalysisFromFirestore = async () => {
  try {
    setLoading(true);
    
    // FIRST: Reset translation state to English immediately
    setSelectedLanguage('en');
    setTranslatedContent(null);
    setTranslationLoading(false);
    
    const analysisData = await getAnalysisById(analysisId, currentUser.uid);
    
    console.log("📦 Loaded analysis from Firestore:", analysisData);
    
    const transformedAnalysis = {
      filename: analysisData.originalFilename,
      file_type: analysisData.fileType,
      summary: analysisData.summary,
      risks: analysisData.risks || [],
      clauses: analysisData.clauses || [],
      total_risky_clauses: analysisData.total_risky_clauses,
      pii_redacted: analysisData.piiRedacted,
      redacted_text: analysisData.redactedDocumentText,
      redacted_document_text: analysisData.redactedDocumentText,
      suggestions: analysisData.suggestions || [],
      fairness_analysis: analysisData.fairness_analysis || [],
      privacy_notice: analysisData.piiRedacted ? "✓ Your Personal Data Has Been Redacted for Privacy." : null,
    };
    
    console.log("✅ Transformed analysis:", transformedAnalysis);
    console.log("📊 Risk count:", transformedAnalysis.risks?.length || 0);
    console.log("📊 Clause count:", transformedAnalysis.clauses?.length || 0);
    
    setAnalysis(transformedAnalysis);
    setAnalysisType(analysisData.analysisType);
    
    setLoading(false);
  } catch (error) {
    console.error("Error loading analysis:", error);
    setLoading(false);
    alert("Failed to load analysis. Redirecting to dashboard...");
    navigate("/dashboard");
  }
};


useEffect(() => {
    const chatContainer = document.getElementById("chat-container");
    if (chatContainer) chatContainer.scrollTop = chatContainer.scrollHeight;
  }, [chatHistory]);

  // Translation handler
  // ============================================
// REPLACE your handleLanguageChange function in Results.jsx
// ============================================
// Fixed handleLanguageChange function
const handleLanguageChange = async (languageCode) => {
  if (languageCode === 'en') {
    setSelectedLanguage('en');
    setTranslatedContent(null);
    return;
  }

  if (!analysisId) {
    console.warn('⚠ No analysisId available');
    alert('This analysis needs to be saved before translation. Please refresh and try again.');
    return;
  }

  if (!currentUser) {
    console.error('❌ No user authenticated');
    alert('Please login to use translation feature');
    return;
  }

  setTranslationLoading(true);
  setSelectedLanguage(languageCode);

  try {
    console.log('🔍 Starting translation process...', {
      analysisId,
      language: languageCode,
      userId: currentUser.uid
    });

    let translation = null;
    
    try {
      translation = await getTranslation(analysisId, languageCode);
      
      if (translation && typeof translation === 'object') {
        console.log('✅ Found cached translation:', translation);
        setTranslatedContent(translation);
        setTranslationLoading(false);
        return;
      }
    } catch (error) {
      console.log('📝 No cached translation found, requesting new one...');
    }

    console.log('🔄 Requesting new translation from backend...');
    translation = await requestTranslation(analysisId, languageCode, currentUser.uid);

    console.log('📦 Backend response:', translation);

    if (!translation) {
      throw new Error('Backend returned null or undefined');
    }

    if (typeof translation !== 'object') {
      throw new Error(`Invalid translation format: expected object, got ${typeof translation}`);
    }

    const hasRequiredFields = translation.summary || translation.risks || translation.clauses;
    
    if (!hasRequiredFields) {
      console.error('❌ Translation missing required fields:', translation);
      throw new Error('Translation is missing required content (summary, risks, or clauses)');
    }

    setTranslatedContent(translation);
    console.log('✅ Translation loaded successfully:', languageCode);
    
  } catch (error) {
    console.error('❌ Translation error:', error);
    console.error('Error stack:', error.stack);
    
    let errorMessage = 'Failed to translate content. ';
    
    if (error.message.includes('Analysis not found')) {
      errorMessage += 'Document not found in database. Try re-analyzing the document.';
    } else if (error.message.includes('Unauthorized')) {
      errorMessage += 'You do not have permission to access this document.';
    } else if (error.message.includes('Firestore not configured')) {
      errorMessage += 'Translation service not available. Please contact support.';
    } else if (error.message.includes('Backend returned null')) {
      errorMessage += 'Backend did not return translation data. This might be a server issue. Please try again or contact support.';
    } else if (error.message.includes('Invalid translation format')) {
      errorMessage += 'Received invalid data from server. Please try again.';
    } else if (error.message.includes('missing required content')) {
      errorMessage += 'Translation data is incomplete. Please try again or contact support.';
    } else {
      errorMessage += error.message || 'Please try again.';
    }
    
    alert(errorMessage);
    setSelectedLanguage('en');
    setTranslatedContent(null);
  } finally {
    setTranslationLoading(false);
  }
};
// Get displayed content based on selected language
  // Get displayed content based on selected language
  const getDisplayedContent = () => {
  // If loading translation, show loading state
  if (translationLoading) {
    return {
      summary: "",
      risks: [],
      clauses: [],
      suggestions: []
    };
  }

  // If English is selected OR no translation exists, show original content
  if (selectedLanguage === 'en' || !translatedContent) {
    return {
      summary: analysis?.summary || "",
      risks: analysis?.risks || [],
      clauses: analysis?.clauses || [],
      suggestions: analysis?.suggestions || []
    };
  }

  // Show translated content if available
  return {
    summary: translatedContent.summary || analysis?.summary || "",
    risks: translatedContent.risks || analysis?.risks || [],
    clauses: translatedContent.clauses || analysis?.clauses || [],
    suggestions: translatedContent.suggestions || analysis?.suggestions || []
  };
};

  const toggleClause = (index) => {
    setExpandedClauses(prev => ({
      ...prev,
      [index]: !prev[index]
    }));
  };

  const handleSendMessage = async () => {
    if (!chatMessage.trim()) return;

    const userMessage = chatMessage.trim();
    setChatMessage("");
    setChatLoading(true);

    setChatHistory((prev) => [...prev, { role: "user", content: userMessage }]);

    try {
      const documentText = analysis.redacted_text || analysis.redacted_document_text || "";
      
      if (!documentText) {
        throw new Error("Document text not available");
      }

      const res = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: userMessage,
          document_text: documentText,
        }),
      });

      const data = await res.json();
      setChatHistory((prev) => [
        ...prev,
        { role: "assistant", content: data.reply || "No response received." },
      ]);
    } catch (error) {
      console.error(error);
      setChatHistory((prev) => [
        ...prev,
        { role: "assistant", content: error.message === "Document text not available" 
          ? "❌ Document text not available. Please re-analyze the document." 
          : "❌ Error communicating with the server." },
      ]);
    } finally {
      setChatLoading(false);
    }
  };

  const handleGenerateNegotiation = async (clauseText) => {
    setSelectedClauseForNegotiation(clauseText);
    setNegotiationLoading(true);
    setNegotiationEmail("");

    try {
      const res = await fetch("http://localhost:8000/draft-negotiation", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ clause: clauseText }),
      });

      const data = await res.json();
      setNegotiationEmail(data.negotiation_email || "Could not generate email.");
    } catch (error) {
      console.error(error);
      setNegotiationEmail("Error generating negotiation email.");
    } finally {
      setNegotiationLoading(false);
    }
  };

  const handleCopyEmail = () => {
    navigator.clipboard.writeText(negotiationEmail);
    setCopiedEmail(true);
    setTimeout(() => setCopiedEmail(false), 2000);
  };

  const handleDraftInGmail = () => {
    if (!negotiationEmail) return;
    
    const subject = encodeURIComponent("Request for Contract Clause Review");
    const body = encodeURIComponent(negotiationEmail);
    const gmailUrl = `https://mail.google.com/mail/?view=cm&fs=1&su=${subject}&body=${body}`;
    
    window.open(gmailUrl, '_blank');
  };

  const handleCloseNegotiation = () => {
    setSelectedClauseForNegotiation(null);
    setNegotiationEmail("");
    setCopiedEmail(false);
  };

  const handleGenerateDocumentEmail = async () => {
    setShowDocumentEmail(true);
    setDocumentEmailLoading(true);
    setDocumentEmail("");
    setEmailSent(false);

    try {
      const displayedContent = getDisplayedContent();
      let documentText = "";
      let riskSummary = "";
      
      if (isDetailedAnalysis) {
        const clauses = displayedContent.clauses || [];
        documentText = displayedContent.summary || "Detailed clause analysis completed";
        riskSummary = `Found ${clauses.length} risky clauses requiring attention.`;
      } else {
        const risks = displayedContent.risks || [];
        documentText = displayedContent.summary || "";
        riskSummary = `Identified ${risks.length} potential risks in the document.`;
      }

      const res = await fetch("http://localhost:8000/draft-document-email", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          document_summary: documentText,
          risk_summary: riskSummary,
        }),
      });

      const data = await res.json();
      setDocumentEmail(data.document_email || data.email || "Email generated successfully.");
    } catch (error) {
      console.error(error);
      setDocumentEmail("Error generating document review. Please try again.");
    } finally {
      setDocumentEmailLoading(false);
    }
  };

  const handleSendDocumentEmail = async () => {
    if (!userEmail || !userEmail.includes('@')) {
      alert('Please enter a valid email address');
      return;
    }

    setEmailSending(true);
    setEmailSent(false);

    try {
      const displayedContent = getDisplayedContent();
      let clauses = [];
      let documentText = "";
      let riskSummary = "";
      
      if (isDetailedAnalysis) {
        clauses = displayedContent.clauses || [];
        documentText = displayedContent.summary || "Detailed clause analysis completed";
        riskSummary = `Found ${clauses.length} risky clauses requiring attention.`;
      } else {
        const risks = displayedContent.risks || [];
        clauses = risks.map(r => ({
          clause: r.clause_text,
          risk: r.severity,
          explanation: r.risk_explanation
        }));
        documentText = displayedContent.summary || "";
        riskSummary = `Identified ${risks.length} potential risks in the document.`;
      }

      const res = await fetch("http://localhost:8000/send-document-review", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          filename: analysis.filename || "Document",
          document_summary: documentText,
          risk_summary: riskSummary,
          clauses: clauses,
          user_email: userEmail,
        }),
      });

      const data = await res.json();
      
      if (data.success) {
        setEmailSent(true);
        setTimeout(() => {
          setShowDocumentEmail(false);
          setUserEmail("");
          setEmailSent(false);
        }, 3000);
      } else {
        alert(data.detail || 'Failed to send email');
      }
    } catch (error) {
      console.error(error);
      alert('Error sending email. Please check your email configuration in backend .env file.');
    } finally {
      setEmailSending(false);
    }
  };

  const handleCopyDocumentEmail = () => {
    navigator.clipboard.writeText(documentEmail);
    setCopiedDocumentEmail(true);
    setTimeout(() => setCopiedDocumentEmail(false), 2000);
  };

  const handleCloseDocumentEmail = () => {
    setShowDocumentEmail(false);
    setDocumentEmail("");
    setCopiedDocumentEmail(false);
  };

  const getRiskBadge = (level) => {
    const styles = {
      High: "bg-red-500/20 text-red-400 border-red-500/50",
      Medium: "bg-yellow-500/20 text-yellow-400 border-yellow-500/50",
      Low: "bg-blue-500/20 text-blue-400 border-blue-500/50",
    };
    return styles[level] || styles.Medium;
  };

  const getLanguageName = (code) => {
    const names = {
      'hi': 'Hindi',
      'es': 'Spanish',
      'fr': 'French',
      'de': 'German',
      'zh': 'Chinese'
    };
    return names[code] || code;
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-black via-[#0F2A40] to-[#064E3B] flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-16 h-16 text-cyan-400 animate-spin mx-auto mb-4" />
          <p className="text-gray-300 text-lg">Loading analysis...</p>
        </div>
      </div>
    );
  }

  if (!analysis) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-black via-[#0F2A40] to-[#064E3B] flex items-center justify-center">
        <div className="text-center">
          <AlertTriangle className="w-16 h-16 text-yellow-400 mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-white mb-4">No Analysis Data</h2>
          <div className="flex gap-4 justify-center">
            <Button onClick={() => navigate("/upload")} className="bg-cyan-600">
              Analyze New Document
            </Button>
            <Button onClick={() => navigate("/dashboard")} variant="outline" className="border-cyan-400 text-cyan-400">
              Go to Dashboard
            </Button>
          </div>
        </div>
      </div>
    );
  }

  // const isDetailedAnalysis = analysisType === "detailed" || (analysis.clauses && Array.isArray(analysis.clauses));
    const isDetailedAnalysis = analysisType === "detailed" || (analysis.clauses && Array.isArray(analysis.clauses) && analysis.clauses.length > 0);

  const displayedContent = getDisplayedContent();
  
  // Render Detailed Clause Analysis View
  if (isDetailedAnalysis) {
    const clauses = displayedContent.clauses || [];
    const totalClauses = analysis.total_risky_clauses || clauses.length;
    const fileType = analysis.file_type || "Text";

    return (
      <div className="min-h-screen relative bg-gradient-to-b from-black via-[#0F2A40] to-[#064E3B] overflow-hidden py-16">
        <div className="absolute inset-0 aurora-bg opacity-20" />

        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          {/* Header with Language Selector */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex items-center justify-between mb-8 flex-wrap gap-4"
          >
            <Button
              onClick={() => navigate("/upload")}
              variant="outline"
              className="border-cyan-400/50 text-cyan-400 hover:bg-cyan-400/10"
            >
              <ArrowLeft className="w-4 h-4 mr-2" />
              New Analysis
            </Button>
            
            <div className="flex items-center gap-4">
              <LanguageSelector
                selectedLanguage={selectedLanguage}
                onLanguageChange={handleLanguageChange}
                loading={translationLoading}
              />
              
              <div className="text-right">
                <h1 className="text-3xl font-bold text-white mb-1">
                  Detailed Clause Analysis
                </h1>
                <p className="text-gray-300 text-sm">
                  {analysis.filename} • {totalClauses} risk{totalClauses !== 1 ? "s" : ""} identified
                </p>
              </div>
            </div>
          </motion.div>

          {/* Privacy Notice */}
          {analysis.privacy_notice && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.05 }}
              className="mb-6 px-4 py-3 bg-green-700/70 text-green-100 rounded-lg font-medium text-sm flex items-center gap-2"
            >
              <CheckCircle className="w-5 h-5 stroke-current flex-shrink-0" />
              <span>{analysis.privacy_notice}</span>
            </motion.div>
          )}

          {/* Translation Indicator */}
          {selectedLanguage !== 'en' && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.06 }}
              className="mb-6 px-4 py-3 bg-blue-700/70 text-blue-100 rounded-lg font-medium text-sm flex items-center gap-2"
            >
              <Globe className="w-5 h-5 stroke-current flex-shrink-0" />
              <span>Content translated to {getLanguageName(selectedLanguage)}</span>
            </motion.div>
          )}

          <div className="grid lg:grid-cols-3 gap-8">
            {/* Main Content - Left/Center */}
            <div className="lg:col-span-2 space-y-6">
              {/* Risk Summary Card */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
              >
                <Card className="border-none bg-gradient-to-br from-red-900/40 to-orange-900/40 backdrop-blur-md shadow-2xl">
                  <CardContent className="p-8">
                    <div className="flex items-start gap-6">
                      <div className="p-4 bg-red-500/20 rounded-xl">
                        <AlertTriangle className="w-10 h-10 text-red-400" />
                      </div>
                      <div className="flex-1">
                        <h2 className="text-2xl font-bold text-white mb-2">
                          Risk Assessment Complete
                        </h2>
                        <p className="text-gray-200 mb-4">
                          We've identified {totalClauses} potentially problematic clause{totalClauses !== 1 ? "s" : ""} that require your attention. Review each one carefully and consider the recommendations provided.
                        </p>
                        <div className="flex gap-4 text-sm flex-wrap">
                          <div className="flex items-center gap-2">
                            <div className="w-3 h-3 rounded-full bg-red-500"></div>
                            <span className="text-gray-300">
                              {clauses.filter((c) => c.risk_level === "High").length} High Risk
                            </span>
                          </div>
                          <div className="flex items-center gap-2">
                            <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
                            <span className="text-gray-300">
                              {clauses.filter((c) => c.risk_level === "Medium").length} Medium Risk
                            </span>
                          </div>
                          <div className="flex items-center gap-2">
                            <div className="w-3 h-3 rounded-full bg-blue-500"></div>
                            <span className="text-gray-300">
                              {clauses.filter((c) => c.risk_level === "Low").length} Low Risk
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>

              {/* Collapsible Clause Cards */}
              <div className="space-y-4">
                {clauses.length > 0 ? (
                  clauses.map((clause, index) => (
                    <motion.div
                      key={index}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.2 + index * 0.05 }}
                    >
                      <Card className="border-none bg-[#064E3B]/90 backdrop-blur-md shadow-xl hover:shadow-2xl transition-all overflow-hidden">
                        <div
                          onClick={() => toggleClause(index)}
                          className="cursor-pointer hover:bg-cyan-700/40 transition-colors"
                        >
                          <CardHeader className="border-b border-gray-700/50 p-6">
                            <div className="flex items-start justify-between gap-4">
                              <div className="flex-1">
                                <div className="flex items-center gap-3 mb-3">
                                  <span className="text-gray-400 font-mono text-sm font-semibold">
                                    Clause #{index + 1}
                                  </span>
                                  <span
                                    className={`px-3 py-1 rounded-full text-xs font-semibold border ${getRiskBadge(
                                      clause.risk_level
                                    )}`}
                                  >
                                    {clause.risk_level} Risk
                                  </span>
                                </div>
                                <p className="text-gray-300 italic text-sm leading-relaxed bg-black/30 p-4 rounded-lg border border-gray-700/50">
                                  "{clause.clause}"
                                </p>
                              </div>
                              <div className="flex-shrink-0 pt-2">
                                {expandedClauses[index] ? (
                                  <ChevronUp className="w-5 h-5 text-cyan-400" />
                                ) : (
                                  <ChevronDown className="w-5 h-5 text-cyan-400" />
                                )}
                              </div>
                            </div>
                          </CardHeader>
                        </div>

                        <AnimatePresence>
                          {expandedClauses[index] && (
                            <motion.div
                              initial={{ height: 0, opacity: 0 }}
                              animate={{ height: "auto", opacity: 1 }}
                              exit={{ height: 0, opacity: 0 }}
                              transition={{ duration: 0.3 }}
                            >
                              <CardContent className="p-6 space-y-6">
                                <div className="bg-red-900/20 border border-red-500/30 rounded-lg p-4">
                                  <div className="flex items-start gap-3">
                                    <AlertTriangle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
                                    <div>
                                      <h4 className="text-white font-semibold mb-1">
                                        Potential Impact
                                      </h4>
                                      <p className="text-gray-200 text-sm leading-relaxed">
                                        {clause.impact}
                                      </p>
                                    </div>
                                  </div>
                                </div>

                                <div className="bg-blue-900/20 border border-blue-500/30 rounded-lg p-4">
                                  <div className="flex items-start gap-3">
                                    <FileText className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
                                    <div>
                                      <h4 className="text-white font-semibold mb-1">
                                        Why This Matters
                                      </h4>
                                      <p className="text-gray-200 text-sm leading-relaxed">
                                        {clause.explanation}
                                      </p>
                                    </div>
                                  </div>
                                </div>

                                <div className="bg-emerald-900/20 border border-emerald-500/30 rounded-lg p-4">
                                  <div className="flex items-start gap-3">
                                    <CheckCircle className="w-5 h-5 text-emerald-400 flex-shrink-0 mt-0.5" />
                                    <div className="flex-1">
                                      <h4 className="text-white font-semibold mb-1">
                                        What You Should Do
                                      </h4>
                                      <p className="text-gray-200 text-sm leading-relaxed">
                                        {clause.recommendation}
                                      </p>
                                    </div>
                                  </div>
                                </div>

                                <div className="pt-2">
                                  <Button
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      handleGenerateNegotiation(clause.clause);
                                    }}
                                    size="sm"
                                    className="bg-gradient-to-r from-cyan-600 to-cyan-500 hover:from-cyan-500 hover:to-cyan-400 text-white shadow-lg"
                                  >
                                    <Mail className="w-4 h-4 mr-2" />
                                    Draft Negotiation Email
                                  </Button>
                                </div>
                              </CardContent>
                            </motion.div>
                          )}
                        </AnimatePresence>
                      </Card>
                    </motion.div>
                  ))
                ) : (
                  <Card className="bg-green-900/30 backdrop-blur-md shadow-2xl border-2 border-green-500/50">
                    <CardHeader>
                      <CardTitle className="flex items-center space-x-2">
                        <CheckCircle className="w-6 h-6 text-green-400" />
                        <span className="text-white">Great News!</span>
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p className="text-gray-200 text-lg">
                        No significant risks detected in this document. The clauses appear to be standard and fair.
                      </p>
                    </CardContent>
                  </Card>
                )}
              </div>
            </div>

            {/* Sidebar - Right Side */}
            <div className="space-y-6">
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.3 }}
              >
                <Card className="border-none bg-[#064E3B]/90 backdrop-blur-md shadow-2xl">
                  <CardHeader className="border-b border-gray-700/50">
                    <CardTitle className="text-white text-lg">Analysis Status</CardTitle>
                  </CardHeader>
                  <CardContent className="p-4 space-y-4">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-200">Processing</span>
                      <Badge variant="secondary">Ready</Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-200">Document Type</span>
                      <Badge variant="outline">{fileType}</Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-200">Analysis Type</span>
                      <Badge variant="outline">Detailed</Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-200">Risk Level</span>
                      <Badge
                        variant={
                          clauses.some((c) => c.risk_level === "High")
                            ? "destructive"
                            : "secondary"
                        }
                      >
                        {clauses.some((c) => c.risk_level === "High") ? "High" : "Medium"}
                      </Badge>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.4 }}
              >
                <Card className="border-none bg-[#064E3B]/90 backdrop-blur-md shadow-2xl">
                  <CardHeader className="border-b border-gray-700/50">
                    <CardTitle className="flex items-center gap-2 text-white text-lg">
                      <Shield className="w-5 h-5 text-purple-400" />
                      Quick Stats
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="p-4 space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-gray-300 text-sm">Total Clauses:</span>
                      <span className="text-white font-bold">{clauses.length}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-300 text-sm">High Risk:</span>
                      <span className="text-red-400 font-bold">
                        {clauses.filter((c) => c.risk_level === "High").length}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-300 text-sm">Medium Risk:</span>
                      <span className="text-yellow-400 font-bold">
                        {clauses.filter((c) => c.risk_level === "Medium").length}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-300 text-sm">Low Risk:</span>
                      <span className="text-blue-400 font-bold">
                        {clauses.filter((c) => c.risk_level === "Low").length}
                      </span>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.5 }}
              >
                <Card className="border-none bg-[#064E3B]/90 backdrop-blur-md shadow-2xl">
                  <CardHeader className="border-b border-gray-700/50">
                    <CardTitle className="text-white text-lg">Key Responsibilities</CardTitle>
                  </CardHeader>
                  <CardContent className="p-4">
                    <div className="space-y-3">
                      {clauses.length > 0 ? (
                        clauses.slice(0, 3).map((clause, idx) => (
                          <div key={idx} className="flex items-start space-x-2">
                            <CheckCircle className="flex-shrink-0 w-4 h-4 text-emerald-400 mt-0.5" />
                            <span className="text-sm text-gray-200 leading-snug">
                              {clause.explanation || "Review clause carefully"}
                            </span>
                          </div>
                        ))
                      ) : (
                        <div className="flex items-start space-x-2">
                          <CheckCircle className="w-4 h-4 text-emerald-400 mt-0.5" />
                          <span className="text-sm text-gray-200">
                            No specific responsibilities identified
                          </span>
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.55 }}
              >
                <Card className="border-none bg-gradient-to-br from-purple-900/40 to-pink-900/40 backdrop-blur-md shadow-2xl">
                  <CardHeader className="border-b border-gray-700/50">
                    <CardTitle className="flex items-center gap-2 text-white text-lg">
                      <Mail className="w-5 h-5 text-pink-400" />
                      Document Email
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="p-4">
                    <p className="text-gray-200 text-sm mb-4">
                      Generate a comprehensive email covering all risks and findings in this document.
                    </p>
                    <Button
                      onClick={handleGenerateDocumentEmail}
                      className="w-full bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 text-white"
                    >
                      <Mail className="w-4 h-4 mr-2" />
                      Generate Document Email
                    </Button>
                  </CardContent>
                </Card>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.6 }}
              >
                <Card className="border-none bg-[#064E3B]/90 backdrop-blur-md shadow-2xl flex flex-col h-[500px]">
                  <CardHeader className="border-b border-gray-700/50">
                    <CardTitle className="flex items-center gap-2 text-white text-lg">
                      <MessageSquare className="w-5 h-5 text-cyan-400" />
                      Chat with Document
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="flex flex-col flex-1 overflow-hidden p-4">
                    <div
                      id="chat-container"
                      className="flex-1 flex flex-col gap-2 overflow-y-auto px-2 py-2"
                      style={{
                        scrollbarWidth: "thin",
                        scrollbarColor: "#0FC6B2 #0F2A40",
                      }}
                    >
                      {chatHistory.map((msg, idx) => (
                        <div
                          key={idx}
                          className={`inline-block max-w-[70%] p-3 rounded-2xl break-words shadow-md transition-shadow duration-200 ${
                            msg.role === "user"
                              ? "bg-gradient-to-r from-cyan-600 to-cyan-500 text-white self-end mr-0"
                              : "bg-gray-800 text-gray-100 self-start ml-0"
                          }`}
                        >
                          {msg.content}
                        </div>
                      ))}
                      {chatLoading && (
                        <p className="text-gray-300 italic text-sm self-start">AI is typing...</p>
                      )}
                    </div>

                    <div className="mt-2 flex space-x-2">
                      <input
                        type="text"
                        value={chatMessage}
                        onChange={(e) => setChatMessage(e.target.value)}
                        onKeyDown={(e) => e.key === "Enter" && handleSendMessage()}
                        placeholder="Type your message..."
                        className="flex-1 p-3 rounded-2xl bg-[#064E3B]/80 text-white border border-gray-600 focus:outline-none placeholder-gray-400"
                      />
                      <Button
                        onClick={handleSendMessage}
                        disabled={chatLoading || !chatMessage.trim()}
                        className="bg-cyan-600 hover:bg-cyan-500 text-white rounded-2xl px-6"
                      >
                        Send
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>

              <Button
                onClick={() => navigate("/upload")}
                className="w-full bg-cyan-600 hover:bg-cyan-500 text-white py-3"
              >
                Analyze Another Document
              </Button>
            </div>
          </div>
        </div>

        {/* Negotiation Email Modal */}
        {selectedClauseForNegotiation && (
          <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 p-4">
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="bg-[#064E3B] rounded-xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto border border-cyan-400/30"
            >
              <div className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-2xl font-bold text-white flex items-center gap-2">
                    <Mail className="w-6 h-6 text-cyan-400" />
                    Negotiation Email
                  </h3>
                  <button
                    onClick={handleCloseNegotiation}
                    className="text-gray-400 hover:text-white transition-colors"
                  >
                    <X className="w-6 h-6" />
                  </button>
                </div>

                <div className="mb-4 bg-black/30 p-4 rounded-lg border border-gray-700">
                  <p className="text-gray-400 text-sm mb-2">Selected Clause:</p>
                  <p className="text-gray-200 italic text-sm">
                    "{selectedClauseForNegotiation}"
                  </p>
                </div>

                {negotiationLoading ? (
                  <div className="flex items-center justify-center py-12">
                    <div className="animate-spin rounded-full h-12 w-12 border-4 border-cyan-400 border-t-transparent"></div>
                  </div>
                ) : negotiationEmail ? (
                  <div>
                    <div className="bg-gray-800 p-6 rounded-lg mb-4 border border-gray-700 max-h-96 overflow-y-auto">
                      <pre className="text-gray-200 whitespace-pre-wrap font-sans text-sm leading-relaxed">
                        {negotiationEmail}
                      </pre>
                    </div>

                    <div className="flex gap-3">
                      <Button
                        onClick={handleDraftInGmail}
                        className="flex-1 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white shadow-lg"
                      >
                        <Mail className="w-4 h-4 mr-2" />
                        Draft in Gmail
                      </Button>
                      <Button
                        onClick={handleCopyEmail}
                        className="flex-1 bg-cyan-600 hover:bg-cyan-500 text-white"
                      >
                        {copiedEmail ? (
                          <>
                            <Check className="w-4 h-4 mr-2" />
                            Copied!
                          </>
                        ) : (
                          <>
                            <Copy className="w-4 h-4 mr-2" />
                            Copy to Clipboard
                          </>
                        )}
                      </Button>
                      <Button
                        onClick={handleCloseNegotiation}
                        variant="outline"
                        className="border-gray-600 text-gray-300 hover:bg-gray-800"
                      >
                        Close
                      </Button>
                    </div>
                  </div>
                ) : null}
              </div>
            </motion.div>
          </div>
        )}

        {/* Document Email Modal */}
        {showDocumentEmail && (
          <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 p-4">
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="bg-gradient-to-br from-purple-900/90 to-pink-900/90 rounded-xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto border border-purple-400/30"
            >
              <div className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-2xl font-bold text-white flex items-center gap-2">
                    <Mail className="w-6 h-6 text-pink-400" />
                    Document Review Email
                  </h3>
                  <button
                    onClick={handleCloseDocumentEmail}
                    className="text-gray-400 hover:text-white transition-colors"
                  >
                    <X className="w-6 h-6" />
                  </button>
                </div>

                <div className="mb-4 bg-black/30 p-4 rounded-lg border border-purple-500/30">
                  <p className="text-gray-300 text-sm">
                    Comprehensive email covering all findings from the document analysis
                  </p>
                </div>

                {documentEmailLoading ? (
                  <div className="flex items-center justify-center py-12">
                    <div className="animate-spin rounded-full h-12 w-12 border-4 border-pink-400 border-t-transparent"></div>
                  </div>
                ) : documentEmail ? (
                  <div>
                    <div className="bg-gray-800 p-6 rounded-lg mb-4 border border-gray-700 max-h-96 overflow-y-auto">
                      <pre className="text-gray-200 whitespace-pre-wrap font-sans text-sm leading-relaxed">
                        {documentEmail}
                      </pre>
                    </div>

                    <div className="mb-4">
                      <label className="block text-gray-300 text-sm font-medium mb-2">
                        Your Email Address
                      </label>
                      <input
                        type="email"
                        value={userEmail}
                        onChange={(e) => setUserEmail(e.target.value)}
                        placeholder="Enter your email to receive PDF report"
                        className="w-full px-4 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-pink-500 transition-colors"
                      />
                    </div>

                    {emailSent && (
                      <div className="mb-4 bg-green-900/30 border border-green-500/50 rounded-lg p-3">
                        <p className="text-green-400 text-sm flex items-center gap-2">
                          <Check className="w-4 h-4" />
                          Email sent successfully! Check your inbox.
                        </p>
                      </div>
                    )}

                    <div className="flex gap-3">
                      <Button
                        onClick={handleSendDocumentEmail}
                        disabled={!userEmail || emailSending}
                        className="flex-1 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 text-white disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        {emailSending ? (
                          <>
                            <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent mr-2"></div>
                            Sending...
                          </>
                        ) : (
                          <>
                            <Mail className="w-4 h-4 mr-2" />
                            Send PDF via Email
                          </>
                        )}
                      </Button>
                      <Button
                        onClick={handleCopyDocumentEmail}
                        className="flex-1 bg-cyan-600 hover:bg-cyan-500 text-white"
                      >
                        {copiedDocumentEmail ? (
                          <>
                            <Check className="w-4 h-4 mr-2" />
                            Copied!
                          </>
                        ) : (
                          <>
                            <Copy className="w-4 h-4 mr-2" />
                            Copy Text
                          </>
                        )}
                      </Button>
                      <Button
                        onClick={handleCloseDocumentEmail}
                        variant="outline"
                        className="border-gray-600 text-gray-300 hover:bg-gray-800"
                      >
                        Close
                      </Button>
                    </div>
                  </div>
                ) : null}
              </div>
            </motion.div>
          </div>
        )}
        <BackToTop />
      </div>
    );
  }

  // Render Standard Analysis View
  const summary = displayedContent.summary;
  const risks = displayedContent.risks;
  const suggestions = displayedContent.suggestions;
  const fairnessAnalysis = analysis.fairness_analysis || [];
  const fileType = analysis.file_type || "Text";

  return (
    <div className="min-h-screen relative bg-gradient-to-b from-black via-[#0F2A40] to-[#064E3B] overflow-hidden py-16">
      <div className="absolute inset-0 aurora-bg opacity-20" />

      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header with Language Selector */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center justify-between mb-8 flex-wrap gap-4"
        >
          <Button
            onClick={() => navigate("/upload")}
            variant="outline"
            className="border-cyan-400/50 text-cyan-400 hover:bg-cyan-400/10"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            New Analysis
          </Button>
          
          <div className="flex items-center gap-4">
            <LanguageSelector
              selectedLanguage={selectedLanguage}
              onLanguageChange={handleLanguageChange}
              loading={translationLoading}
            />
            
            <div className="text-right">
              <h1 className="text-3xl font-bold text-white mb-1">
                Standard Analysis Results
              </h1>
              <p className="text-gray-300 text-sm">
                {analysis.filename || "Document"} • {risks.length} risk{risks.length !== 1 ? "s" : ""} identified
              </p>
            </div>
          </div>
        </motion.div>

        {/* Privacy Notice */}
        {analysis.privacy_notice && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.05 }}
            className="mb-6 px-4 py-3 bg-green-700/70 text-green-100 rounded-lg font-medium text-sm flex items-center gap-2"
          >
            <CheckCircle className="w-5 h-5 stroke-current flex-shrink-0" />
            <span>{analysis.privacy_notice}</span>
          </motion.div>
        )}

        {/* Translation Indicator */}
        {selectedLanguage !== 'en' && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.06 }}
            className="mb-6 px-4 py-3 bg-blue-700/70 text-blue-100 rounded-lg font-medium text-sm flex items-center gap-2"
          >
            <Globe className="w-5 h-5 stroke-current flex-shrink-0" />
            <span>Content translated to {getLanguageName(selectedLanguage)}</span>
          </motion.div>
        )}

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Main Content - Left Side */}
          <div className="lg:col-span-2 space-y-6">
            {/* Summary Card */}
            {/* Summary Card */}
<motion.div
  initial={{ opacity: 0, y: 20 }}
  animate={{ opacity: 1, y: 0 }}
  transition={{ delay: 0.1 }}
>
  <Card className="border-none bg-[#064E3B]/90 backdrop-blur-md shadow-2xl">
    <CardHeader className="border-b border-gray-700/50">
      <CardTitle className="flex items-center gap-2 text-white">
        <FileText className="w-5 h-5 text-cyan-400" />
        Document Summary
      </CardTitle>
    </CardHeader>
    <CardContent className="p-6">
      <MarkdownRenderer text={summary} />
    </CardContent>
  </Card>
</motion.div>
            {/* Risks Card */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
            >
              <Card className="border-none bg-[#064E3B]/90 backdrop-blur-md shadow-2xl">
                <CardHeader className="border-b border-gray-700/50">
                  <CardTitle className="flex items-center gap-2 text-white">
                    <AlertTriangle className="w-5 h-5 text-red-400" />
                    Identified Risks ({risks.length})
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-6">
                  {risks.length === 0 ? (
                    <div className="flex items-center gap-3 text-emerald-400 bg-emerald-900/20 p-4 rounded-lg border border-emerald-500/30">
                      <CheckCircle className="w-5 h-5" />
                      <span>No significant risks detected in this document.</span>
                    </div>
                  ) : (
                    <div className="space-y-4">
                      {risks.map((risk, index) => (
                        <div
                          key={index}
                          className="bg-gray-800/50 border border-gray-700 rounded-lg p-4 hover:border-cyan-400/50 transition-all"
                        >
                          <div className="flex items-start justify-between gap-4 mb-3">
                            <div className="flex-1">
                              <div className="flex items-center gap-2 mb-2">
                                <span
                                  className={`px-3 py-1 rounded-full text-xs font-semibold border ${getRiskBadge(
                                    risk.severity
                                  )}`}
                                >
                                  {risk.severity} Risk
                                </span>
                              </div>
                              <p className="text-gray-300 text-sm italic mb-2 bg-black/30 p-3 rounded border border-gray-700/50">
                                "{risk.clause_text}"
                              </p>
                              <p className="text-gray-200 text-sm">
                                {risk.risk_explanation}
                              </p>
                            </div>
                          </div>
                          
                          <Button
                            onClick={() => handleGenerateNegotiation(risk.clause_text)}
                            size="sm"
                            className="bg-cyan-600 hover:bg-cyan-500 text-white mt-2"
                          >
                            <Mail className="w-4 h-4 mr-2" />
                            Draft Negotiation Email
                          </Button>
                        </div>
                      ))}
                    </div>
                  )}
                </CardContent>
              </Card>
            </motion.div>

            {/* Fairness Analysis */}
            {fairnessAnalysis.length > 0 && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
              >
                <Card className="border-none bg-[#064E3B]/90 backdrop-blur-md shadow-2xl">
                  <CardHeader className="border-b border-gray-700/50">
                    <CardTitle className="flex items-center gap-2 text-white">
                      <Scale className="w-5 h-5 text-purple-400" />
                      Fairness Analysis
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="p-6 space-y-6">
                    {fairnessAnalysis.map((item, index) => (
                      <div key={index} className="border border-gray-700 rounded-lg p-4 bg-gray-800/30">
                        <div className="mb-4">
                          <div className="flex items-center gap-3 mb-2">
                            <span className="text-sm font-semibold text-gray-400">Fairness Score:</span>
                            <div className="flex items-center gap-2">
                              <div className="w-32 h-2 bg-gray-700 rounded-full overflow-hidden">
                                <div
                                  className={`h-full ${
                                    item.fairness_score >= 70
                                      ? "bg-emerald-500"
                                      : item.fairness_score >= 40
                                      ? "bg-yellow-500"
                                      : "bg-red-500"
                                  }`}
                                  style={{ width: `${item.fairness_score}%` }}
                                />
                              </div>
                              <span className="text-white font-bold">{item.fairness_score}/100</span>
                            </div>
                          </div>
                        </div>

                        <div className="space-y-3">
                          <div>
                            <h4 className="text-sm font-semibold text-red-400 mb-1">Risky Clause:</h4>
                            <p className="text-gray-300 text-sm italic bg-red-900/20 p-2 rounded border border-red-500/30">
                              "{item.risky_clause}"
                            </p>
                          </div>

                          <div>
                            <h4 className="text-sm font-semibold text-emerald-400 mb-1">Suggested Standard Clause:</h4>
                            <p className="text-gray-300 text-sm bg-emerald-900/20 p-2 rounded border border-emerald-500/30">
                              {item.standard_clause}
                            </p>
                          </div>

                          <div>
                            <h4 className="text-sm font-semibold text-blue-400 mb-1">Explanation:</h4>
                            <p className="text-gray-200 text-sm">{item.explanation}</p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </CardContent>
                </Card>
              </motion.div>
            )}

            {/* Suggestions Card */}
            {suggestions.length > 0 && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 }}
              >
                <Card className="border-none bg-[#064E3B]/90 backdrop-blur-md shadow-2xl">
                  <CardHeader className="border-b border-gray-700/50">
                    <CardTitle className="flex items-center gap-2 text-white">
                      <TrendingUp className="w-5 h-5 text-emerald-400" />
                      Suggested Actions
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="p-6">
                    <div className="space-y-3">
                      {suggestions.map((suggestion, index) => (
                        <div
                          key={index}
                          className="flex items-start gap-3 bg-emerald-900/20 border border-emerald-500/30 p-4 rounded-lg"
                        >
                          <CheckCircle className="w-5 h-5 text-emerald-400 flex-shrink-0 mt-0.5" />
                          <p className="text-gray-200 text-sm">{highlightText(suggestion)}</p>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            )}
          </div>

          {/* Sidebar - Right Side */}
          <div className="space-y-6">
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 }}
            >
              <Card className="border-none bg-[#064E3B]/90 backdrop-blur-md shadow-2xl">
                <CardHeader className="border-b border-gray-700/50">
                  <CardTitle className="text-white text-lg">Analysis Status</CardTitle>
                </CardHeader>
                <CardContent className="p-4 space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-200">Processing</span>
                    <Badge variant="secondary">Complete</Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-200">Document Type</span>
                    <Badge variant="outline">{fileType}</Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-200">Analysis Type</span>
                    <Badge variant="outline">Standard</Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-200">Risk Level</span>
                    <Badge
                      variant={
                        risks.some((r) => r.severity === "High")
                          ? "destructive"
                          : "secondary"
                      }
                    >
                      {risks.some((r) => r.severity === "High") ? "High" : risks.length > 0 ? "Medium" : "Low"}
                    </Badge>
                  </div>
                </CardContent>
              </Card>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.4 }}
            >
              <Card className="border-none bg-[#064E3B]/90 backdrop-blur-md shadow-2xl">
                <CardHeader className="border-b border-gray-700/50">
                  <CardTitle className="flex items-center gap-2 text-white text-lg">
                    <Shield className="w-5 h-5 text-purple-400" />
                    Quick Stats
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-4 space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-gray-300 text-sm">Total Risks:</span>
                    <span className="text-white font-bold">{risks.length}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-300 text-sm">High Risk:</span>
                    <span className="text-red-400 font-bold">
                      {risks.filter((r) => r.severity === "High").length}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-300 text-sm">Medium Risk:</span>
                    <span className="text-yellow-400 font-bold">
                      {risks.filter((r) => r.severity === "Medium").length}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-300 text-sm">Suggestions:</span>
                    <span className="text-emerald-400 font-bold">{suggestions.length}</span>
                  </div>
                </CardContent>
              </Card>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.5 }}
            >
              <Card className="border-none bg-[#064E3B]/90 backdrop-blur-md shadow-2xl">
                <CardHeader className="border-b border-gray-700/50">
                  <CardTitle className="text-white text-lg">Key Responsibilities</CardTitle>
                </CardHeader>
                <CardContent className="p-4">
                  <div className="space-y-3">
                    {risks.length > 0 ? (
                      risks.slice(0, 3).map((risk, idx) => (
                        <div key={idx} className="flex items-start space-x-2">
                          <CheckCircle className="flex-shrink-0 w-4 h-4 text-emerald-400 mt-0.5" />
                          <span className="text-sm text-gray-200 leading-snug">
                            {risk.risk_explanation || "Review this risk carefully"}
                          </span>
                        </div>
                      ))
                    ) : (
                      <div className="flex items-start space-x-2">
                        <CheckCircle className="w-4 h-4 text-emerald-400 mt-0.5" />
                        <span className="text-sm text-gray-200">
                          No specific responsibilities identified
                        </span>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.55 }}
            >
              <Card className="border-none bg-gradient-to-br from-purple-900/40 to-pink-900/40 backdrop-blur-md shadow-2xl">
                <CardHeader className="border-b border-gray-700/50">
                  <CardTitle className="flex items-center gap-2 text-white text-lg">
                    <Mail className="w-5 h-5 text-pink-400" />
                    Document Email
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-4">
                  <p className="text-gray-200 text-sm mb-4">
                    Generate a comprehensive email covering all risks and findings in this document.
                  </p>
                  <Button
                    onClick={handleGenerateDocumentEmail}
                    className="w-full bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 text-white"
                  >
                    <Mail className="w-4 h-4 mr-2" />
                    Generate Document Email
                  </Button>
                </CardContent>
              </Card>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.6 }}
            >
              <Card className="border-none bg-[#064E3B]/90 backdrop-blur-md shadow-2xl flex flex-col h-[500px]">
                <CardHeader className="border-b border-gray-700/50">
                  <CardTitle className="flex items-center gap-2 text-white text-lg">
                    <MessageSquare className="w-5 h-5 text-cyan-400" />
                    Chat with Document
                  </CardTitle>
                </CardHeader>
                <CardContent className="flex flex-col flex-1 overflow-hidden p-4">
                  <div
                    id="chat-container"
                    className="flex-1 flex flex-col gap-2 overflow-y-auto px-2 py-2"
                    style={{
                      scrollbarWidth: "thin",
                      scrollbarColor: "#0FC6B2 #0F2A40",
                    }}
                  >
                    {chatHistory.map((msg, idx) => (
                      <div
                        key={idx}
                        className={`inline-block max-w-[70%] p-3 rounded-2xl break-words shadow-md transition-shadow duration-200 ${
                          msg.role === "user"
                            ? "bg-gradient-to-r from-cyan-600 to-cyan-500 text-white self-end mr-0"
                            : "bg-gray-800 text-gray-100 self-start ml-0"
                        }`}
                      >
                        {msg.content}
                      </div>
                    ))}
                    {chatLoading && (
                      <p className="text-gray-300 italic text-sm self-start">AI is typing...</p>
                    )}
                  </div>

                  <div className="mt-2 flex space-x-2">
                    <input
                      type="text"
                      value={chatMessage}
                      onChange={(e) => setChatMessage(e.target.value)}
                      onKeyDown={(e) => e.key === "Enter" && handleSendMessage()}
                      placeholder="Type your message..."
                      className="flex-1 p-3 rounded-2xl bg-[#064E3B]/80 text-white border border-gray-600 focus:outline-none placeholder-gray-400"
                    />
                    <Button
                      onClick={handleSendMessage}
                      disabled={chatLoading || !chatMessage.trim()}
                      className="bg-cyan-600 hover:bg-cyan-500 text-white rounded-2xl px-6"
                    >
                      Send
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </motion.div>

            <Button
              onClick={() => navigate("/upload")}
              className="w-full bg-cyan-600 hover:bg-cyan-500 text-white py-3"
            >
              Analyze Another Document
            </Button>
          </div>
        </div>
      </div>

      {/* Negotiation Email Modal */}
      {selectedClauseForNegotiation && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-[#064E3B] rounded-xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto border border-cyan-400/30"
          >
            <div className="p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-2xl font-bold text-white flex items-center gap-2">
                  <Mail className="w-6 h-6 text-cyan-400" />
                  Negotiation Email
                </h3>
                <button
                  onClick={handleCloseNegotiation}
                  className="text-gray-400 hover:text-white transition-colors"
                >
                  <X className="w-6 h-6" />
                </button>
              </div>

              <div className="mb-4 bg-black/30 p-4 rounded-lg border border-gray-700">
                <p className="text-gray-400 text-sm mb-2">Selected Clause:</p>
                <p className="text-gray-200 italic text-sm">
                  "{selectedClauseForNegotiation}"
                </p>
              </div>

              {negotiationLoading ? (
                <div className="flex items-center justify-center py-12">
                  <div className="animate-spin rounded-full h-12 w-12 border-4 border-cyan-400 border-t-transparent"></div>
                </div>
              ) : negotiationEmail ? (
                <div>
                  <div className="bg-gray-800 p-6 rounded-lg mb-4 border border-gray-700 max-h-96 overflow-y-auto">
                    <pre className="text-gray-200 whitespace-pre-wrap font-sans text-sm leading-relaxed">
                      {negotiationEmail}
                    </pre>
                  </div>

                  <div className="flex gap-3">
                    <Button
                      onClick={handleDraftInGmail}
                      className="flex-1 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white shadow-lg"
                    >
                      <Mail className="w-4 h-4 mr-2" />
                      Draft in Gmail
                    </Button>
                    <Button
                      onClick={handleCopyEmail}
                      className="flex-1 bg-cyan-600 hover:bg-cyan-500 text-white"
                    >
                      {copiedEmail ? (
                        <>
                          <Check className="w-4 h-4 mr-2" />
                          Copied!
                        </>
                      ) : (
                        <>
                          <Copy className="w-4 h-4 mr-2" />
                          Copy to Clipboard
                        </>
                      )}
                    </Button>
                    <Button
                      onClick={handleCloseNegotiation}
                      variant="outline"
                      className="border-gray-600 text-gray-300 hover:bg-gray-800"
                    >
                      Close
                    </Button>
                  </div>
                </div>
              ) : null}
            </div>
          </motion.div>
        </div>
      )}

      {/* Document Email Modal */}
      {showDocumentEmail && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-gradient-to-br from-purple-900/90 to-pink-900/90 rounded-xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto border border-purple-400/30"
          >
            <div className="p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-2xl font-bold text-white flex items-center gap-2">
                  <Mail className="w-6 h-6 text-pink-400" />
                  Document Review Email
                </h3>
                <button
                  onClick={handleCloseDocumentEmail}
                  className="text-gray-400 hover:text-white transition-colors"
                >
                  <X className="w-6 h-6" />
                </button>
              </div>

              <div className="mb-4 bg-black/30 p-4 rounded-lg border border-purple-500/30">
                <p className="text-gray-300 text-sm">
                  Comprehensive email covering all findings from the document analysis
                </p>
              </div>

              {documentEmailLoading ? (
                <div className="flex items-center justify-center py-12">
                  <div className="animate-spin rounded-full h-12 w-12 border-4 border-pink-400 border-t-transparent"></div>
                </div>
              ) : documentEmail ? (
                <div>
                  <div className="bg-gray-800 p-6 rounded-lg mb-4 border border-gray-700 max-h-96 overflow-y-auto">
                    <pre className="text-gray-200 whitespace-pre-wrap font-sans text-sm leading-relaxed">
                      {documentEmail}
                    </pre>
                  </div>

                  <div className="flex gap-3">
                    <Button
                      onClick={handleCopyDocumentEmail}
                      className="flex-1 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 text-white"
                    >
                      {copiedDocumentEmail ? (
                        <>
                          <Check className="w-4 h-4 mr-2" />
                          Copied!
                        </>
                      ) : (
                        <>
                          <Copy className="w-4 h-4 mr-2" />
                          Copy to Clipboard
                        </>
                      )}
                    </Button>
                    <Button
                      onClick={handleCloseDocumentEmail}
                      variant="outline"
                      className="border-gray-600 text-gray-300 hover:bg-gray-800"
                    >
                      Close
                    </Button>
                  </div>
                </div>
              ) : null}
            </div>
          </motion.div>
        </div>
      )}
      <BackToTop />
    </div>
  );
}