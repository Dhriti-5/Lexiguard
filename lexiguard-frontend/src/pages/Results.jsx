import React from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Badge } from "../components/ui/badge";
import { FileText, AlertTriangle, CheckCircle } from "lucide-react";
import { Button } from "../components/ui/button";
import FairnessScore from "../components/FairnessScore";
import NegotiationAssistant from "../components/NegotiationAssistant";

const KEYWORDS_REGEX = /\b(agreement|contract|clause|liability|termination|penalty|indemnity|renewal)\b/gi;
const ACTIONABLE_REGEX = /\b(\d+\s*(days?|weeks?|months?|years?)|due by|deadline|penalty|fine|termination|payment of \$?\d+|effective date|must|shall|obligated|required)\b/gi;

const backendURL = process.env.REACT_APP_BACKEND_URL;

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
  const initialAnalysis = location.state?.analysis;

  const [analysis, setAnalysis] = React.useState(initialAnalysis);
  const [chatMessages, setChatMessages] = React.useState([]);
  const [inputMessage, setInputMessage] = React.useState("");
  const [loading, setLoading] = React.useState(false);
  const [loadingAnalysis, setLoadingAnalysis] = React.useState(false);

  // Auto-scroll chat
  React.useEffect(() => {
    const chatContainer = document.getElementById("chat-container");
    if (chatContainer) chatContainer.scrollTop = chatContainer.scrollHeight;
  }, [chatMessages]);

  // Fetch extended analysis if fairness is missing
  React.useEffect(() => {
    if (initialAnalysis && !initialAnalysis.fairness_analysis) {
      const fetchExtended = async () => {
        setLoadingAnalysis(true);
        try {
          const response = await fetch(`${backendURL}/analyze-extended`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: initialAnalysis.summary || initialAnalysis.text || "" }),
        });

        if (!response.ok) {
          console.error("Failed to fetch extended analysis");
          setLoadingAnalysis(false);
          return;
        }

        const data = await response.json();
        setAnalysis(data);
        } catch (error) {
          console.error("Error fetching extended analysis:", error);
        } finally {
          setLoadingAnalysis(false);
        }
      };
      fetchExtended();
    }
  }, [initialAnalysis]);

const handleSend = async () => {
  if (!inputMessage.trim()) return;

  const userMessage = inputMessage.trim();
  setChatMessages(prev => [...prev, { sender: "user", text: userMessage }]);
  setInputMessage("");
  setLoading(true);

  try {
    const response = await fetch(`${backendURL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: userMessage,
        document_text: analysis.summary || analysis.text || "",
      }),
    });

    if (!response.ok) {
      throw new Error(`Server responded with status ${response.status}`);
    }

    const data = await response.json();
    const aiMessage = data.reply || "No answer available.";
    setChatMessages(prev => [...prev, { sender: "ai", text: aiMessage }]);
  } catch (error) {
    console.error("Chat error:", error);
    setChatMessages(prev => [
      ...prev,
      { sender: "ai", text: "Error communicating with server." },
    ]);
  } finally {
    setLoading(false);
  }
};


  if (!analysis) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-b from-black via-[#0F2A40] to-[#064E3B] text-white">
        <h2 className="text-3xl font-bold mb-4">No Analysis Data</h2>
        <p className="mb-6 text-gray-300">Please go back and provide a legal document or text.</p>
        <Button onClick={() => navigate("/upload")}>Go Back</Button>
      </div>
    );
  }

  const fileType = analysis.file_type || "Text";

  return (
    <div className="min-h-screen py-16 bg-gradient-to-b from-black via-[#0F2A40] to-[#064E3B] text-white">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl md:text-5xl font-bold mb-2">Analysis Results</h1>
          <p className="text-lg md:text-xl text-gray-300 mb-1">
            Your AI-powered legal document analysis is ready
          </p>
          <Badge variant="outline" className="text-white border-gray-500">
            Detected File Type: {fileType.toUpperCase()}
          </Badge>
          {analysis.privacy_notice && (
            <div className="mt-4 px-4 py-3 bg-green-700/70 text-green-100 rounded-lg font-medium text-sm flex items-center gap-2 max-w-6xl mx-auto">
              <CheckCircle className="w-5 h-5 stroke-current" />
              <span>{analysis.privacy_notice}</span>
            </div>
          )}
        </div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Left Column */}
          <div className="lg:col-span-2 space-y-6">
            {/* Summary */}
            <Card className="bg-[#064E3B]/90 backdrop-blur-md shadow-2xl border-none">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <FileText className="w-5 h-5 text-cyan-400" />
                  <span>Document Summary</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="text-gray-200 whitespace-pre-line leading-relaxed">
                {highlightText(analysis.summary || "")}
              </CardContent>
            </Card>

            {/* Risks */}
            <Card className="bg-[#064E3B]/90 backdrop-blur-md shadow-2xl border-none">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <AlertTriangle className="w-5 h-5 text-red-400" />
                  <span>Risk Analysis</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                {analysis.risks && analysis.risks.length > 0 ? (
                  <ul className="space-y-4">
                    {analysis.risks.map((risk, idx) => (
                      <li key={idx} className="border border-gray-700 rounded-lg p-4 bg-[#1E1E1E]">
                        <p><strong>Clause:</strong> {risk.clause_text || "N/A"}</p>
                        <p><strong>Reason:</strong> {risk.risk_explanation || "No reason provided."}</p>
                        <p>
                          <strong>Severity:</strong>{" "}
                          <Badge
                            variant={risk.severity === "High" ? "destructive" : "secondary"}
                          >
                            {risk.severity || "Unknown"}
                          </Badge>
                        </p>
                        <div className="mt-3">
                          <NegotiationAssistant riskyClause={risk.clause_text} />
                        </div>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-gray-300">No risks detected in this document.</p>
                )}
              </CardContent>
            </Card>

            {/* Suggestions */}
            <Card className="bg-[#064E3B]/90 backdrop-blur-md shadow-2xl border-none">
              <CardHeader>
                <CardTitle>Suggested Actions</CardTitle>
              </CardHeader>
              <CardContent>
                {(analysis.suggestions || []).length > 0 ? (
                  <ul className="space-y-3 list-disc list-inside text-gray-200">
                    {analysis.suggestions.map((suggestion, idx) => (
                      <li key={idx}>{highlightText(suggestion || "")}</li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-gray-300">No specific suggestions provided.</p>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Right Column */}
          <div className="space-y-6">
            {/* Status */}
            <Card className="bg-[#064E3B]/90 backdrop-blur-md shadow-2xl border-none">
              <CardHeader>
                <CardTitle>Analysis Status</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-200">Processing</span>
                  <Badge variant="secondary">Ready</Badge>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-200">Document Type</span>
                  <Badge variant="outline">{fileType}</Badge>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-200">Risk Level</span>
                  <Badge
                    variant={
                      analysis.risks && analysis.risks.some((r) => r.severity === "High")
                        ? "destructive"
                        : "secondary"
                    }
                  >
                    {analysis.risks && analysis.risks.some((r) => r.severity === "High")
                      ? "High"
                      : "Medium"}
                  </Badge>
                </div>
              </CardContent>
            </Card>

            {/* Fairness Score */}
            {loadingAnalysis ? (
              <p className="text-gray-300 italic">Loading fairness analysis...</p>
            ) : (
              analysis.fairness_analysis &&
              analysis.fairness_analysis.length > 0 && (
                <Card className="bg-[#064E3B]/90 backdrop-blur-md shadow-2xl border-none">
                  <CardHeader>
                    <CardTitle>Fairness Score</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <FairnessScore data={analysis.fairness_analysis} />
                  </CardContent>
                </Card>
              )
            )}

<Card className="bg-[#064E3B]/90 backdrop-blur-md shadow-2xl border-none">
  <CardHeader>
    <CardTitle>Key Responsibilities</CardTitle>
  </CardHeader>
  <CardContent>
    <div className="space-y-3">
      {analysis.risks && analysis.risks.length > 0 ? (
        analysis.risks.map((risk, idx) => (
          <div key={idx} className="flex items-center space-x-2">
            <CheckCircle className="flex-shrink-0 w-4 h-4 text-emerald-400" />
            <span className="text-sm text-gray-200 leading-snug">
              {risk.risk_explanation || "No responsibility specified."}
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


            {/* Negotiation Assistant */}
            <Card className="bg-[#064E3B]/90 backdrop-blur-md shadow-2xl border-none">
              <CardHeader>
                <CardTitle>Negotiation Assistant</CardTitle>
              </CardHeader>
              <CardContent>
                <NegotiationAssistant riskyClause={analysis.summary || ""} />
              </CardContent>
            </Card>

            {/* Chat */}
            <Card className="bg-[#064E3B]/90 backdrop-blur-md shadow-2xl border-none flex flex-col h-[500px]">
              <CardHeader>
                <CardTitle>Chat with Document</CardTitle>
              </CardHeader>
              <CardContent className="flex flex-col flex-1 overflow-hidden">
                <div
                  id="chat-container"
                  className="flex-1 flex flex-col gap-2 overflow-y-auto px-2 py-2"
                  style={{
                    scrollbarWidth: "thin",
                    scrollbarColor: "#0FC6B2 #0F2A40",
                  }}
                >
                  {chatMessages.map((msg, idx) => (
                    <div
                      key={idx}
                      className={`inline-block max-w-[70%] p-3 rounded-2xl break-words shadow-md transition-shadow duration-200 ${
                        msg.sender === "user"
                          ? "bg-gradient-to-r from-cyan-600 to-cyan-500 text-white self-end mr-0"
                          : "bg-gray-800 text-gray-100 self-start ml-0"
                      }`}
                    >
                      {msg.text}
                    </div>
                  ))}
                  {loading && (
                    <p className="text-gray-300 italic text-sm self-start">AI is typing...</p>
                  )}
                  <style jsx>{`
                    #chat-container::-webkit-scrollbar {
                      width: 4px;
                    }
                    #chat-container::-webkit-scrollbar-track {
                      background: #0F2A40;
                    }
                    #chat-container::-webkit-scrollbar-thumb {
                      background-color: #0FC6B2;
                      border-radius: 10px;
                    }
                  `}</style>
                </div>

                <div className="mt-2 flex space-x-2">
                  <input
                    type="text"
                    value={inputMessage}
                    onChange={(e) => setInputMessage(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && handleSend()}
                    placeholder="Type your message..."
                    className="flex-1 p-3 rounded-2xl bg-[#064E3B]/80 text-white border border-gray-600 focus:outline-none placeholder-gray-400"
                  />
                  <Button
                    onClick={handleSend}
                    disabled={loading}
                    className="bg-cyan-600 hover:bg-cyan-500 text-white rounded-2xl px-6"
                  >
                    Send
                  </Button>
                </div>
              </CardContent>
            </Card>

            <Button
              onClick={() => navigate("/upload")}
              className="w-full bg-cyan-600 text-white py-3"
            >
              Analyze Another Document
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
