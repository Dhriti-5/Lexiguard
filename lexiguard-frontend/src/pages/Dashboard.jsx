import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { 
  FileText, 
  AlertTriangle, 
  Shield, 
  Clock, 
  Star, 
  Trash2, 
  Edit2, 
  Eye,
  CheckCircle,
  X,
  Search,
  TrendingUp,
  BarChart3,
  Loader2,
  FolderOpen
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import {
  getUserAnalyses,
  deleteAnalysis,
  updateAnalysisTitle,
  toggleStarredAnalysis,
  getAnalysisStats
} from "../services/firestoreService";

export default function Dashboard() {
  const navigate = useNavigate();
  const { currentUser } = useAuth();
  
  const [analyses, setAnalyses] = useState([]);
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [filterType, setFilterType] = useState("all");
  const [sortBy, setSortBy] = useState("date");
  
  const [editingId, setEditingId] = useState(null);
  const [editTitle, setEditTitle] = useState("");
  const [deletingId, setDeletingId] = useState(null);

  useEffect(() => {
    if (currentUser) {
      loadDashboardData();
    }
  }, [currentUser]);

  const loadDashboardData = async () => {
    try {
      setLoading(true);
      const [userAnalyses, userStats] = await Promise.all([
        getUserAnalyses(currentUser.uid),
        getAnalysisStats(currentUser.uid)
      ]);
      setAnalyses(userAnalyses);
      setStats(userStats);
    } catch (error) {
      console.error("Error loading dashboard:", error);
      alert("Failed to load dashboard data");
    } finally {
      setLoading(false);
    }
  };

  const handleViewAnalysis = (analysisId) => {
    navigate(`/results/${analysisId}`);
  };

  const handleStartEdit = (analysis) => {
    setEditingId(analysis.id);
    setEditTitle(analysis.documentTitle);
  };

  const handleSaveEdit = async (analysisId) => {
    try {
      await updateAnalysisTitle(analysisId, currentUser.uid, editTitle);
      setAnalyses(analyses.map(a => 
        a.id === analysisId ? { ...a, documentTitle: editTitle } : a
      ));
      setEditingId(null);
    } catch (error) {
      console.error("Error updating title:", error);
      alert("Failed to update title");
    }
  };

  const handleCancelEdit = () => {
    setEditingId(null);
    setEditTitle("");
  };

  const handleDeleteConfirm = async () => {
    try {
      await deleteAnalysis(deletingId, currentUser.uid);
      setAnalyses(analyses.filter(a => a.id !== deletingId));
      setDeletingId(null);
      
      const userStats = await getAnalysisStats(currentUser.uid);
      setStats(userStats);
    } catch (error) {
      console.error("Error deleting analysis:", error);
      alert("Failed to delete analysis");
    }
  };

  const handleToggleStar = async (analysisId, currentStarred) => {
    try {
      await toggleStarredAnalysis(analysisId, currentUser.uid, !currentStarred);
      setAnalyses(analyses.map(a => 
        a.id === analysisId ? { ...a, starred: !currentStarred } : a
      ));
    } catch (error) {
      console.error("Error toggling star:", error);
    }
  };

  const getFilteredAnalyses = () => {
    let filtered = analyses;

    if (searchTerm) {
      filtered = filtered.filter(a => 
        a.documentTitle.toLowerCase().includes(searchTerm.toLowerCase()) ||
        a.originalFilename.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    if (filterType === "starred") {
      filtered = filtered.filter(a => a.starred);
    } else if (filterType !== "all") {
      filtered = filtered.filter(a => a.analysisType === filterType);
    }

    if (sortBy === "date") {
      filtered.sort((a, b) => b.uploadTimestamp - a.uploadTimestamp);
    } else if (sortBy === "title") {
      filtered.sort((a, b) => a.documentTitle.localeCompare(b.documentTitle));
    } else if (sortBy === "risk") {
      filtered.sort((a, b) => {
        const getRiskCount = (analysis) => {
          if (analysis.analysisType === 'detailed') {
            return analysis.clauses?.filter(c => c.risk_level === 'High').length || 0;
          }
          return analysis.risks?.filter(r => r.severity === 'High').length || 0;
        };
        return getRiskCount(b) - getRiskCount(a);
      });
    }

    return filtered;
  };

  const filteredAnalyses = getFilteredAnalyses();

  const getRiskBadge = (analysis) => {
    let highCount = 0;
    let totalCount = 0;

    if (analysis.analysisType === 'detailed' && analysis.clauses) {
      highCount = analysis.clauses.filter(c => c.risk_level === 'High').length;
      totalCount = analysis.clauses.length;
    } else if (analysis.risks) {
      highCount = analysis.risks.filter(r => r.severity === 'High').length;
      totalCount = analysis.risks.length;
    }

    if (highCount > 0) {
      return { color: "bg-red-500/20 text-red-400 border-red-500/50", text: `${highCount} High Risk` };
    } else if (totalCount > 0) {
      return { color: "bg-yellow-500/20 text-yellow-400 border-yellow-500/50", text: `${totalCount} Risk${totalCount > 1 ? 's' : ''}` };
    }
    return { color: "bg-green-500/20 text-green-400 border-green-500/50", text: "Low Risk" };
  };

  // Helper function to safely render summary text
  const renderSummary = (summary) => {
    if (!summary) return null;
    
    // If summary is a string, return it
    if (typeof summary === 'string') {
      return summary;
    }
    
    // If summary is an object (shouldn't happen, but handle it)
    if (typeof summary === 'object') {
      return JSON.stringify(summary);
    }
    
    return null;
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-black via-[#0F2A40] to-[#064E3B] flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-16 h-16 text-cyan-400 animate-spin mx-auto mb-4" />
          <p className="text-gray-300 text-lg">Loading your dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen relative bg-gradient-to-b from-black via-[#0F2A40] to-[#064E3B] overflow-hidden py-16">
      <div className="absolute inset-0 aurora-bg opacity-20" />

      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            <div>
              <h1 className="text-4xl md:text-5xl font-bold text-white mb-2">
                Your Legal Dashboard
              </h1>
              <p className="text-lg text-gray-300">
                Securely manage and review your document analyses
              </p>
            </div>
            <Button
              onClick={() => navigate("/upload")}
              className="bg-gradient-to-r from-cyan-600 to-cyan-500 hover:from-cyan-500 hover:to-cyan-400 text-white shadow-lg"
            >
              <FileText className="w-4 h-4 mr-2" />
              Analyze New Document
            </Button>
          </div>
        </motion.div>

        {/* Privacy Notice */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="mb-8 bg-green-900/30 border border-green-500/50 rounded-lg p-4"
        >
          <div className="flex items-start gap-3">
            <Shield className="w-5 h-5 text-green-400 flex-shrink-0 mt-0.5" />
            <div>
              <h3 className="text-green-400 font-semibold mb-1">Privacy Protected</h3>
              <p className="text-gray-200 text-sm">
                All stored analyses contain only <strong>redacted</strong> document text with PII replaced by placeholders. 
                Your original documents are <strong>never stored</strong> - only the privacy-safe analysis results.
              </p>
            </div>
          </div>
        </motion.div>

        {/* Stats Cards */}
        {stats && (
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-8">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
            >
              <Card className="border-none bg-gradient-to-br from-cyan-900/40 to-blue-900/40 backdrop-blur-md shadow-xl">
                <CardContent className="p-6 text-center">
                  <FileText className="w-8 h-8 text-cyan-400 mx-auto mb-2" />
                  <div className="text-3xl font-bold text-white mb-1">{stats.totalAnalyses}</div>
                  <div className="text-xs text-gray-300">Total Analyses</div>
                </CardContent>
              </Card>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.25 }}
            >
              <Card className="border-none bg-gradient-to-br from-red-900/40 to-orange-900/40 backdrop-blur-md shadow-xl">
                <CardContent className="p-6 text-center">
                  <AlertTriangle className="w-8 h-8 text-red-400 mx-auto mb-2" />
                  <div className="text-3xl font-bold text-white mb-1">{stats.highRiskCount}</div>
                  <div className="text-xs text-gray-300">High Risk</div>
                </CardContent>
              </Card>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
            >
              <Card className="border-none bg-gradient-to-br from-yellow-900/40 to-amber-900/40 backdrop-blur-md shadow-xl">
                <CardContent className="p-6 text-center">
                  <TrendingUp className="w-8 h-8 text-yellow-400 mx-auto mb-2" />
                  <div className="text-3xl font-bold text-white mb-1">{stats.mediumRiskCount}</div>
                  <div className="text-xs text-gray-300">Medium Risk</div>
                </CardContent>
              </Card>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.35 }}
            >
              <Card className="border-none bg-gradient-to-br from-green-900/40 to-emerald-900/40 backdrop-blur-md shadow-xl">
                <CardContent className="p-6 text-center">
                  <CheckCircle className="w-8 h-8 text-green-400 mx-auto mb-2" />
                  <div className="text-3xl font-bold text-white mb-1">{stats.lowRiskCount}</div>
                  <div className="text-xs text-gray-300">Low Risk</div>
                </CardContent>
              </Card>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
            >
              <Card className="border-none bg-gradient-to-br from-purple-900/40 to-pink-900/40 backdrop-blur-md shadow-xl">
                <CardContent className="p-6 text-center">
                  <BarChart3 className="w-8 h-8 text-purple-400 mx-auto mb-2" />
                  <div className="text-3xl font-bold text-white mb-1">{stats.totalClauses}</div>
                  <div className="text-xs text-gray-300">Total Clauses</div>
                </CardContent>
              </Card>
            </motion.div>
          </div>
        )}

        {/* Search and Filters */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.45 }}
          className="mb-6"
        >
          <Card className="border-none bg-[#064E3B]/90 backdrop-blur-md shadow-xl">
            <CardContent className="p-4">
              <div className="flex flex-col md:flex-row gap-4">
                <div className="flex-1 relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                  <input
                    type="text"
                    placeholder="Search by title or filename..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="w-full pl-10 pr-4 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-cyan-500"
                  />
                </div>

                <select
                  value={filterType}
                  onChange={(e) => setFilterType(e.target.value)}
                  className="px-4 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-cyan-500"
                >
                  <option value="all">All Types</option>
                  <option value="standard">Standard Analysis</option>
                  <option value="detailed">Detailed Analysis</option>
                  <option value="starred">Starred</option>
                </select>

                <select
                  value={sortBy}
                  onChange={(e) => setSortBy(e.target.value)}
                  className="px-4 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-cyan-500"
                >
                  <option value="date">Sort by Date</option>
                  <option value="title">Sort by Title</option>
                  <option value="risk">Sort by Risk Level</option>
                </select>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Analyses Grid */}
        {filteredAnalyses.length === 0 ? (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
          >
            <Card className="border-none bg-[#064E3B]/90 backdrop-blur-md shadow-xl">
              <CardContent className="p-12 text-center">
                <FolderOpen className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-white mb-2">
                  {searchTerm || filterType !== "all" ? "No analyses found" : "No analyses yet"}
                </h3>
                <p className="text-gray-300 mb-6">
                  {searchTerm || filterType !== "all" 
                    ? "Try adjusting your filters or search terms" 
                    : "Upload your first legal document to get started"}
                </p>
                {!searchTerm && filterType === "all" && (
                  <Button
                    onClick={() => navigate("/upload")}
                    className="bg-cyan-600 hover:bg-cyan-500 text-white"
                  >
                    <FileText className="w-4 h-4 mr-2" />
                    Analyze Your First Document
                  </Button>
                )}
              </CardContent>
            </Card>
          </motion.div>
        ) : (
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            <AnimatePresence>
              {filteredAnalyses.map((analysis, index) => {
                const riskBadge = getRiskBadge(analysis);
                
                return (
                  <motion.div
                    key={analysis.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, scale: 0.9 }}
                    transition={{ delay: 0.5 + index * 0.05 }}
                  >
                    <Card className="border-none bg-[#064E3B]/90 backdrop-blur-md shadow-xl hover:shadow-2xl transition-all group">
                      <CardHeader className="border-b border-gray-700/50 pb-4">
                        <div className="flex items-start justify-between gap-2">
                          <div className="flex-1 min-w-0">
                            {editingId === analysis.id ? (
                              <div className="flex items-center gap-2">
                                <input
                                  type="text"
                                  value={editTitle}
                                  onChange={(e) => setEditTitle(e.target.value)}
                                  className="flex-1 px-2 py-1 bg-gray-800 border border-cyan-500 rounded text-white text-sm"
                                  autoFocus
                                />
                                <button
                                  onClick={() => handleSaveEdit(analysis.id)}
                                  className="p-1 text-green-400 hover:text-green-300"
                                >
                                  <CheckCircle className="w-4 h-4" />
                                </button>
                                <button
                                  onClick={handleCancelEdit}
                                  className="p-1 text-red-400 hover:text-red-300"
                                >
                                  <X className="w-4 h-4" />
                                </button>
                              </div>
                            ) : (
                              <CardTitle className="text-white text-lg mb-2 truncate flex items-center gap-2">
                                {analysis.documentTitle}
                                <button
                                  onClick={() => handleStartEdit(analysis)}
                                  className="opacity-0 group-hover:opacity-100 transition-opacity"
                                >
                                  <Edit2 className="w-3 h-3 text-cyan-400 hover:text-cyan-300" />
                                </button>
                              </CardTitle>
                            )}
                            <p className="text-gray-400 text-xs truncate">
                              Original: {analysis.originalFilename}
                            </p>
                          </div>
                          
                          <button
                            onClick={() => handleToggleStar(analysis.id, analysis.starred)}
                            className="flex-shrink-0"
                          >
                            <Star 
                              className={`w-5 h-5 ${
                                analysis.starred 
                                  ? 'fill-yellow-400 text-yellow-400' 
                                  : 'text-gray-400 hover:text-yellow-400'
                              } transition-colors`}
                            />
                          </button>
                        </div>
                      </CardHeader>

                      <CardContent className="p-4 space-y-4">
                        {/* Badges */}
                        <div className="flex flex-wrap gap-2">
                          <Badge className="bg-cyan-500/20 text-cyan-400 border border-cyan-500/50">
                            {analysis.fileType}
                          </Badge>
                          <Badge className="bg-purple-500/20 text-purple-400 border border-purple-500/50">
                            {analysis.analysisType === 'detailed' ? 'Detailed' : 'Standard'}
                          </Badge>
                          <Badge className={`border ${riskBadge.color}`}>
                            {riskBadge.text}
                          </Badge>
                          {analysis.piiRedacted && (
                            <Badge className="bg-green-500/20 text-green-400 border border-green-500/50">
                              ✓ PII Redacted
                            </Badge>
                          )}
                        </div>

                        {/* Date */}
                        <div className="flex items-center gap-2 text-gray-300 text-sm">
                          <Clock className="w-4 h-4" />
                          <span>
                            {new Date(analysis.uploadTimestamp).toLocaleDateString('en-US', {
                              year: 'numeric',
                              month: 'short',
                              day: 'numeric',
                              hour: '2-digit',
                              minute: '2-digit'
                            })}
                          </span>
                        </div>

                        {/* Summary Preview - FIXED */}
                        {analysis.summary && (
                          <p className="text-gray-300 text-sm line-clamp-2">
                            {renderSummary(analysis.summary)}
                          </p>
                        )}

                        {/* Action Buttons */}
                        <div className="flex gap-2 pt-2">
                          <Button
                            onClick={() => handleViewAnalysis(analysis.id)}
                            className="flex-1 bg-cyan-600 hover:bg-cyan-500 text-white"
                          >
                            <Eye className="w-4 h-4 mr-2" />
                            View Analysis
                          </Button>
                          <Button
                            onClick={() => setDeletingId(analysis.id)}
                            variant="outline"
                            className="border-red-500/50 text-red-400 hover:bg-red-500/10"
                          >
                            <Trash2 className="w-4 h-4" />
                          </Button>
                        </div>
                      </CardContent>
                    </Card>
                  </motion.div>
                );
              })}
            </AnimatePresence>
          </div>
        )}
      </div>

      {/* Delete Confirmation Modal */}
      {deletingId && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-[#064E3B] rounded-xl shadow-2xl max-w-md w-full border border-red-500/30"
          >
            <div className="p-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="p-3 bg-red-500/20 rounded-full">
                  <AlertTriangle className="w-6 h-6 text-red-400" />
                </div>
                <h3 className="text-xl font-bold text-white">Delete Analysis?</h3>
              </div>

              <p className="text-gray-300 mb-6">
                Are you sure you want to delete this analysis? This action cannot be undone.
              </p>

              <div className="flex gap-3">
                <Button
                  onClick={handleDeleteConfirm}
                  className="flex-1 bg-red-600 hover:bg-red-500 text-white"
                >
                  Delete
                </Button>
                <Button
                  onClick={() => setDeletingId(null)}
                  variant="outline"
                  className="flex-1 border-gray-600 text-gray-300 hover:bg-gray-800"
                >
                  Cancel
                </Button>
              </div>
            </div>
          </motion.div>
        </div>
      )}
    </div>
  );
}