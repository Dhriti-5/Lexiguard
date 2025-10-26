import React, { useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import {
  Upload as UploadIcon,
  FileText,
  Clock,
  Shield,
  AlertTriangle,
  CheckCircle,
  Zap,
} from "lucide-react";
import { motion } from "framer-motion";
import { saveAnalysis } from "../services/firestoreService";

export default function Upload() {
  const [file, setFile] = useState(null);
  const [textInput, setTextInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [analysisType, setAnalysisType] = useState("detailed");
  const [processingMode, setProcessingMode] = useState("sync");
  const navigate = useNavigate();
  const fileInputRef = useRef(null);
  const { currentUser } = useAuth();

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      const validExtensions = ['pdf', 'docx', 'txt'];
      const fileExtension = selectedFile.name.split('.').pop().toLowerCase();
      
      if (!validExtensions.includes(fileExtension)) {
        alert('Please upload only PDF, DOCX, or TXT files');
        return;
      }
      
      setFile(selectedFile);
    }
  };

  // ✅ FIXED: Async upload with proper error handling
  const handleAsyncAnalyze = async () => {
    if (!file) {
      alert("Please upload a file for async processing");
      return;
    }

    if (!currentUser) {
      alert("Please login to use async processing");
      return;
    }

    setLoading(true);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("documentTitle", file.name);
    formData.append("analysisType", analysisType);
    formData.append("userId", currentUser.uid);

    try {
      console.log('📤 Uploading file for async processing...');
      
      const res = await fetch("http://localhost:8000/analyze-file-async", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || `HTTP ${res.status}`);
      }

      const data = await res.json();

      if (!data.success || !data.jobId) {
        throw new Error(data.detail || "Upload failed - no job ID returned");
      }

      console.log('✅ Async upload successful:', data);
      console.log('   Job ID:', data.jobId);
      console.log('   Status:', data.status);

      setLoading(false);

      // Navigate to results page with job ID for real-time tracking
      navigate(`/results/job/${data.jobId}`, { 
        state: { 
          jobId: data.jobId,
          documentTitle: data.documentTitle || file.name,
          analysisType: analysisType,
          isAsyncJob: true,
          userId: currentUser.uid
        } 
      });
    } catch (error) {
      console.error('❌ Async upload error:', error);
      setLoading(false);
      alert(`Upload failed: ${error.message}`);
    }
  };

  // EXISTING: Sync upload handler (unchanged)
  const handleAnalyze = async () => {
    if (!file && !textInput.trim()) {
      alert("Please provide text or upload a file");
      return;
    }

    setLoading(true);

    const formData = new FormData();
    if (file) {
      formData.append("file", file);
    } else {
      formData.append("text", textInput);
    }

    try {
      const endpoint =
        analysisType === "detailed"
          ? "http://localhost:8000/analyze-clauses"
          : "http://localhost:8000/analyze-file";

      const res = await fetch(endpoint, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      if (data.error) {
        setLoading(false);
        alert(data.error);
        return;
      }

      let analysisId = null;
      if (currentUser) {
        try {
          analysisId = await saveAnalysis(currentUser.uid, data);
          console.log('✅ Analysis saved to dashboard with ID:', analysisId);
        } catch (error) {
          console.error('⚠ Failed to save to dashboard:', error);
        }
      }

      setLoading(false);

      navigate("/results" + (analysisId ? `/${analysisId}` : ""), { 
        state: { analysis: data, analysisType, analysisId } 
      });
    } catch (error) {
      console.error(error);
      setLoading(false);
      alert("Error analyzing document");
    }
  };

  const handleSubmit = () => {
    if (processingMode === "async") {
      handleAsyncAnalyze();
    } else {
      handleAnalyze();
    }
  };

  return (
    <div className="min-h-screen relative bg-gradient-to-b from-black via-[#0F2A40] to-[#064E3B] overflow-hidden py-16">
      <div className="absolute inset-0 aurora-bg opacity-20" />

      <div className="relative max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-4xl md:text-5xl font-bold text-white mb-4"
          >
            Upload Your Legal Document
          </motion.h1>
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="text-lg md:text-xl text-gray-200/70"
          >
            Get instant AI-powered analysis and insights in seconds
          </motion.p>
        </div>

        <Card className="border-none bg-[#064E3B]/90 backdrop-blur-md mb-12 shadow-2xl">
          <CardHeader>
            <CardTitle className="text-center text-2xl text-white">
              Document Upload or Text Input
            </CardTitle>
          </CardHeader>
          <CardContent className="p-10">
            {/* Analysis Type Selector */}
            <div className="mb-8">
              <label className="block text-white text-lg font-semibold mb-4">
                Choose Analysis Type:
              </label>
              <div className="grid md:grid-cols-2 gap-4">
                <div
                  onClick={() => setAnalysisType("standard")}
                  className={`cursor-pointer rounded-xl p-6 border-2 transition-all ${
                    analysisType === "standard"
                      ? "border-cyan-400 bg-cyan-900/30 shadow-lg shadow-cyan-500/50"
                      : "border-gray-600 bg-gray-800/30 hover:border-gray-500"
                  }`}
                >
                  <div className="flex items-start gap-3">
                    <FileText className="w-6 h-6 text-cyan-400 flex-shrink-0 mt-1" />
                    <div>
                      <h3 className="text-white font-bold text-lg mb-2">
                        Standard Analysis
                      </h3>
                      <p className="text-gray-300 text-sm leading-relaxed">
                        Quick summary with basic risk identification and suggestions
                      </p>
                      <div className="mt-3 flex items-center gap-2">
                        <Clock className="w-4 h-4 text-emerald-400" />
                        <span className="text-xs text-emerald-400 font-semibold">
                          ~15-20 seconds
                        </span>
                      </div>
                    </div>
                    {analysisType === "standard" && (
                      <CheckCircle className="w-6 h-6 text-cyan-400 ml-auto" />
                    )}
                  </div>
                </div>

                <div
                  onClick={() => setAnalysisType("detailed")}
                  className={`cursor-pointer rounded-xl p-6 border-2 transition-all ${
                    analysisType === "detailed"
                      ? "border-cyan-400 bg-cyan-900/30 shadow-lg shadow-cyan-500/50"
                      : "border-gray-600 bg-gray-800/30 hover:border-gray-500"
                  }`}
                >
                  <div className="flex items-start gap-3">
                    <AlertTriangle className="w-6 h-6 text-yellow-400 flex-shrink-0 mt-1" />
                    <div>
                      <h3 className="text-white font-bold text-lg mb-2 flex items-center gap-2">
                        Detailed Clause Analysis
                        <span className="text-xs bg-yellow-500 text-black px-2 py-0.5 rounded-full font-bold">
                          NEW
                        </span>
                      </h3>
                      <p className="text-gray-300 text-sm leading-relaxed">
                        Deep dive into each risky clause with explanations, impact, and recommendations
                      </p>
                      <div className="mt-3 flex items-center gap-2">
                        <Clock className="w-4 h-4 text-emerald-400" />
                        <span className="text-xs text-emerald-400 font-semibold">
                          ~20-30 seconds
                        </span>
                      </div>
                    </div>
                    {analysisType === "detailed" && (
                      <CheckCircle className="w-6 h-6 text-cyan-400 ml-auto" />
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* Processing Mode Selector (only for file uploads) */}
            {file && currentUser && (
              <div className="mb-8 p-6 bg-purple-900/20 border-2 border-purple-500/30 rounded-xl">
                <label className="block text-white text-lg font-semibold mb-4">
                  Choose Processing Mode:
                </label>
                <div className="grid md:grid-cols-2 gap-4">
                  <div
                    onClick={() => setProcessingMode("sync")}
                    className={`cursor-pointer rounded-xl p-4 border-2 transition-all ${
                      processingMode === "sync"
                        ? "border-emerald-400 bg-emerald-900/30 shadow-lg shadow-emerald-500/50"
                        : "border-gray-600 bg-gray-800/30 hover:border-gray-500"
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      <Clock className="w-5 h-5 text-emerald-400" />
                      <div className="flex-1">
                        <h4 className="text-white font-bold">Instant</h4>
                        <p className="text-gray-300 text-xs">Wait for results</p>
                      </div>
                      {processingMode === "sync" && (
                        <CheckCircle className="w-5 h-5 text-emerald-400" />
                      )}
                    </div>
                  </div>

                  <div
                    onClick={() => setProcessingMode("async")}
                    className={`cursor-pointer rounded-xl p-4 border-2 transition-all ${
                      processingMode === "async"
                        ? "border-purple-400 bg-purple-900/30 shadow-lg shadow-purple-500/50"
                        : "border-gray-600 bg-gray-800/30 hover:border-gray-500"
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      <Zap className="w-5 h-5 text-purple-400" />
                      <div className="flex-1">
                        <h4 className="text-white font-bold flex items-center gap-2">
                          Background
                          <span className="text-xs bg-purple-500 text-white px-1.5 py-0.5 rounded">
                            BETA
                          </span>
                        </h4>
                        <p className="text-gray-300 text-xs">Upload & go</p>
                      </div>
                      {processingMode === "async" && (
                        <CheckCircle className="w-5 h-5 text-purple-400" />
                      )}
                    </div>
                  </div>
                </div>
                <p className="text-gray-400 text-xs mt-3">
                  {processingMode === "async" 
                    ? "✨ Upload instantly and check back later for results. Perfect for large documents!"
                    : "⏱ Standard processing - you'll wait while analysis completes."}
                </p>
              </div>
            )}

            {/* Text Input */}
            <textarea
              value={textInput}
              onChange={(e) => setTextInput(e.target.value)}
              placeholder="Or paste your legal text here..."
              rows={6}
              className="w-full p-4 border border-gray-700 rounded-lg bg-[#1E1E1E] text-[#EAEAEA] placeholder-gray-500 backdrop-blur-md mb-6"
            />

            {/* File Upload */}
            <div
              className="border-2 border-dashed border-cyan-400/50 rounded-xl p-12 text-center cursor-pointer mb-6 hover:border-cyan-400 transition-colors"
              onClick={() => fileInputRef.current.click()}
            >
              <UploadIcon className="w-16 h-16 text-cyan-400 mx-auto mb-4 animate-pulse" />
              <h3 className="text-xl font-semibold text-white mb-2">
                Drop your document here or click to browse
              </h3>
              <p className="text-gray-200/70 mb-2">
                Supports PDF, Word (DOCX), and Text (TXT) files
              </p>
              <p className="text-sm text-gray-400">
                Maximum file size: 10MB
              </p>
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                className="hidden"
                accept=".pdf,.docx,.txt"
              />
              <Button className="bg-gradient-to-r from-[#0F2A40] to-[#064E3B] text-white px-8 py-3 mt-4">
                Select Document
              </Button>
              {file && (
                <div className="mt-4 flex items-center justify-center gap-2 bg-green-900/30 border border-green-500/50 rounded-lg p-3">
                  <CheckCircle className="w-5 h-5 text-green-400" />
                  <p className="text-green-200 font-medium">{file.name}</p>
                  <span className="text-xs text-green-300 ml-2">
                    ({file.type || 'text/plain'})
                  </span>
                </div>
              )}
            </div>

            {/* Analyze Button */}
            <Button
              onClick={handleSubmit}
              className="bg-gradient-to-r from-cyan-600 to-cyan-500 hover:from-cyan-500 hover:to-cyan-400 text-white w-full py-4 text-lg font-semibold shadow-lg shadow-cyan-500/50 transition-all"
              disabled={loading}
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <svg
                    className="animate-spin h-5 w-5 text-white"
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    ></circle>
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    ></path>
                  </svg>
                  {processingMode === "async" ? "Uploading..." : "Analyzing Document..."}
                </span>
              ) : (
                `${processingMode === "async" ? "Upload & Process in Background" : `Analyze Document (${analysisType === "detailed" ? "Detailed" : "Standard"})`}`
              )}
            </Button>
          </CardContent>
        </Card>

        {/* Feature Cards */}
        <div className="grid md:grid-cols-3 gap-6">
          <Card className="border-none bg-[#064E3B]/80 backdrop-blur-md shadow-xl">
            <CardContent className="p-6 text-center">
              <FileText className="w-12 h-12 text-cyan-400 mx-auto mb-4 animate-pulse" />
              <h3 className="font-semibold text-white mb-2">Smart Analysis</h3>
              <p className="text-gray-200/70 text-sm">
                AI extracts key information and creates easy-to-read summaries
              </p>
            </CardContent>
          </Card>

          <Card className="border-none bg-[#064E3B]/80 backdrop-blur-md shadow-xl">
            <CardContent className="p-6 text-center">
              <Clock className="w-12 h-12 text-emerald-400 mx-auto mb-4 animate-pulse" />
              <h3 className="font-semibold text-white mb-2">Fast Results</h3>
              <p className="text-gray-200/70 text-sm">
                Get comprehensive analysis in under 30 seconds
              </p>
            </CardContent>
          </Card>

          <Card className="border-none bg-[#064E3B]/80 backdrop-blur-md shadow-xl">
            <CardContent className="p-6 text-center">
              <Shield className="w-12 h-12 text-purple-400 mx-auto mb-4 animate-pulse" />
              <h3 className="font-semibold text-white mb-2">Secure & Private</h3>
              <p className="text-gray-200/70 text-sm">
                Your documents are encrypted and never stored permanently
              </p>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
