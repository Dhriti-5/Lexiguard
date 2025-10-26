import React, { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";
import { getAnalysisById } from "../services/firestoreService";
import RoleAwareChatAgent from "../components/RoleAwareChatAgent";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { ArrowLeft, FileText, Loader2, AlertTriangle, MessageSquare } from "lucide-react";
import { motion } from "framer-motion";
import BackToTop from "../components/BackToTop";

/**
 * Standalone Chat with Document page
 * 
 * Accessible from:
 * - Dashboard "Chat" button for each analysis
 * - Direct URL: /chat/:analysisId
 * 
 * Features:
 * - Full-screen chat experience
 * - Document context sidebar
 * - Role-aware intelligent conversations
 */
export default function ChatWithDocument() {
  const { analysisId } = useParams();
  const navigate = useNavigate();
  const { currentUser } = useAuth();
  
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (analysisId && currentUser) {
      loadAnalysis();
    }
  }, [analysisId, currentUser]);

  const loadAnalysis = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const analysisData = await getAnalysisById(analysisId, currentUser.uid);
      setAnalysis(analysisData);
      
      console.log('✅ Analysis loaded for chat:', analysisData.documentTitle);
    } catch (error) {
      console.error('❌ Error loading analysis for chat:', error);
      setError(error.message || 'Failed to load document');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-black via-[#0F2A40] to-[#064E3B] flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-16 h-16 text-cyan-400 animate-spin mx-auto mb-4" />
          <p className="text-gray-300 text-lg">Loading document...</p>
        </div>
      </div>
    );
  }

  if (error || !analysis) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-black via-[#0F2A40] to-[#064E3B] flex items-center justify-center">
        <div className="text-center max-w-md">
          <AlertTriangle className="w-16 h-16 text-yellow-400 mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-white mb-4">Document Not Found</h2>
          <p className="text-gray-300 mb-6">{error || 'The requested document could not be loaded.'}</p>
          <div className="flex gap-4 justify-center">
            <Button onClick={() => navigate("/dashboard")} className="bg-cyan-600">
              Back to Dashboard
            </Button>
            <Button onClick={() => navigate("/upload")} variant="outline" className="border-cyan-400 text-cyan-400">
              Upload New Document
            </Button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen relative bg-gradient-to-b from-black via-[#0F2A40] to-[#064E3B] overflow-hidden">
      <div className="absolute inset-0 aurora-bg opacity-20" />

      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center justify-between mb-8"
        >
          <Button
            onClick={() => navigate("/dashboard")}
            variant="outline"
            className="border-cyan-400/50 text-cyan-400 hover:bg-cyan-400/10"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Dashboard
          </Button>
          
          <div className="text-center">
            <h1 className="text-3xl font-bold text-white mb-2 flex items-center justify-center gap-2">
              <MessageSquare className="w-8 h-8 text-cyan-400" />
              Chat with Document
            </h1>
            <p className="text-gray-300">{analysis.documentTitle}</p>
          </div>
          
          <Button
            onClick={() => navigate(`/results/${analysisId}`)}
            variant="outline"
            className="border-cyan-400/50 text-cyan-400 hover:bg-cyan-400/10"
          >
            <FileText className="w-4 h-4 mr-2" />
            View Analysis
          </Button>
        </motion.div>

        <div className="grid lg:grid-cols-4 gap-8 h-[calc(100vh-200px)]">
          {/* Chat Area - Main */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 }}
            className="lg:col-span-3"
          >
            <RoleAwareChatAgent
              analysisId={analysisId}
              redactedDocumentText={analysis.redactedDocumentText || ''}
              height="100%"
              showTitle={false}
              className="h-full"
            />
          </motion.div>

          {/* Document Info Sidebar */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
            className="lg:col-span-1 space-y-6"
          >
            {/* Document Overview */}
            <Card className="border-none bg-[#064E3B]/90 backdrop-blur-md shadow-2xl">
              <CardHeader className="border-b border-gray-700/50">
                <CardTitle className="text-white text-lg flex items-center gap-2">
                  <FileText className="w-5 h-5 text-cyan-400" />
                  Document Info
                </CardTitle>
              </CardHeader>
              <CardContent className="p-4 space-y-3">
                <div>
                  <span className="text-gray-400 text-sm">Title:</span>
                  <p className="text-white font-medium">{analysis.documentTitle}</p>
                </div>
                <div>
                  <span className="text-gray-400 text-sm">Type:</span>
                  <p className="text-white font-medium">{analysis.fileType}</p>
                </div>
                <div>
                  <span className="text-gray-400 text-sm">Analysis:</span>
                  <p className="text-white font-medium">
                    {analysis.analysisType === 'detailed' ? 'Detailed Clause Analysis' : 'Standard Analysis'}
                  </p>
                </div>
                <div>
                  <span className="text-gray-400 text-sm">Uploaded:</span>
                  <p className="text-white font-medium">
                    {new Date(analysis.uploadTimestamp).toLocaleDateString()}
                  </p>
                </div>
              </CardContent>
            </Card>

            {/* Quick Stats */}
            <Card className="border-none bg-[#064E3B]/90 backdrop-blur-md shadow-2xl">
              <CardHeader className="border-b border-gray-700/50">
                <CardTitle className="text-white text-lg">Quick Stats</CardTitle>
              </CardHeader>
              <CardContent className="p-4 space-y-3">
                {analysis.analysisType === 'detailed' ? (
                  <>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Total Clauses:</span>
                      <span className="text-white font-bold">{analysis.clauses?.length || 0}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">High Risk:</span>
                      <span className="text-red-400 font-bold">
                        {analysis.clauses?.filter(c => c.risk_level === 'High').length || 0}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Medium Risk:</span>
                      <span className="text-yellow-400 font-bold">
                        {analysis.clauses?.filter(c => c.risk_level === 'Medium').length || 0}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Low Risk:</span>
                      <span className="text-blue-400 font-bold">
                        {analysis.clauses?.filter(c => c.risk_level === 'Low').length || 0}
                      </span>
                    </div>
                  </>
                ) : (
                  <>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Total Risks:</span>
                      <span className="text-white font-bold">{analysis.risks?.length || 0}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">High Risk:</span>
                      <span className="text-red-400 font-bold">
                        {analysis.risks?.filter(r => r.severity === 'High').length || 0}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Medium Risk:</span>
                      <span className="text-yellow-400 font-bold">
                        {analysis.risks?.filter(r => r.severity === 'Medium').length || 0}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Suggestions:</span>
                      <span className="text-emerald-400 font-bold">{analysis.suggestions?.length || 0}</span>
                    </div>
                  </>
                )}
              </CardContent>
            </Card>

            {/* Chat Tips */}
            <Card className="border-none bg-gradient-to-br from-purple-900/40 to-pink-900/40 backdrop-blur-md shadow-2xl">
              <CardHeader className="border-b border-gray-700/50">
                <CardTitle className="text-white text-lg">Chat Tips</CardTitle>
              </CardHeader>
              <CardContent className="p-4">
                <div className="space-y-3 text-sm">
                  <div className="text-gray-300">
                    • Tell me your role for personalized insights
                  </div>
                  <div className="text-gray-300">
                    • Ask about specific clauses or terms
                  </div>
                  <div className="text-gray-300">
                    • Request risk explanations
                  </div>
                  <div className="text-gray-300">
                    • Get negotiation suggestions
                  </div>
                  <div className="text-gray-300">
                    • Ask "What should I be worried about?"
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Actions */}
            <div className="space-y-3">
              <Button
                onClick={() => navigate(`/results/${analysisId}`)}
                className="w-full bg-cyan-600 hover:bg-cyan-500"
              >
                <FileText className="w-4 h-4 mr-2" />
                View Full Analysis
              </Button>
              <Button
                onClick={() => navigate("/dashboard")}
                variant="outline"
                className="w-full border-cyan-400/50 text-cyan-400 hover:bg-cyan-400/10"
              >
                Back to Dashboard
              </Button>
            </div>
          </motion.div>
        </div>
      </div>

      <BackToTop />
    </div>
  );
}
