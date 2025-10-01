import React, { useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Upload as UploadIcon, FileText, Clock, Shield } from "lucide-react";
import { motion } from "framer-motion";

export default function Upload() {
  const [file, setFile] = useState(null);
  const [textInput, setTextInput] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

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
      const res = await fetch(`${process.env.REACT_APP_BACKEND_URL}/analyze-file`, {
      method: "POST",
      body: formData,
    });

    // Check if response is OK
    if (!res.ok) {
      const errorText = await res.text(); // Get error from backend
      alert("Server error: " + errorText); // Show error to user
      setLoading(false); // Stop loading spinner
      return; // Exit function
    }

    // Proceed normally
    const data = await res.json();

      setLoading(false);

      if (data.error) {
        alert(data.error);
        return;
      }

      navigate("/results", { state: { analysis: data } });
    } catch (error) {
      console.error(error);
      setLoading(false);
      alert("Error analyzing document");
    }
  };

  return (
    <div className="min-h-screen relative bg-gradient-to-b from-black via-[#0F2A40] to-[#064E3B] overflow-hidden py-16">
      {/* Aurora Glow Background */}
      <div className="absolute inset-0 aurora-bg opacity-20" />

      <div className="relative max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
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

        {/* Upload Card */}
        <Card className="border-none bg-[#064E3B]/90 backdrop-blur-md mb-12 shadow-2xl">
          <CardHeader>
            <CardTitle className="text-center text-2xl text-white">
              Document Upload or Text Input
            </CardTitle>
          </CardHeader>
          <CardContent className="p-10">
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
              className="border-2 border-dashed border-cyan-400/50 rounded-xl p-12 text-center cursor-pointer mb-6"
              onClick={() => fileInputRef.current.click()}
            >
              <UploadIcon className="w-16 h-16 text-cyan-400 mx-auto mb-4 animate-pulse" />
              <h3 className="text-xl font-semibold text-white mb-2">
                Drop your document here or click to browse
              </h3>
              <p className="text-gray-200/70 mb-6">
                Supports PDF, Word, and text files up to 10MB
              </p>
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                className="hidden"
              />
              <Button className="bg-gradient-to-r from-[#0F2A40] to-[#064E3B] text-white px-8 py-3 mt-4">
                Select Document
              </Button>
              {file && <p className="text-gray-200 mt-2">{file.name}</p>}
            </div>

            {/* Analyze Button */}
            <Button
              onClick={handleAnalyze}
              className="bg-cyan-600 text-white w-full py-3"
              disabled={loading}
            >
              {loading ? "Analyzing..." : "Analyze Document"}
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
