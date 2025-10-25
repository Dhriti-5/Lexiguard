import React, { useState, useEffect } from 'react';
import { useParams, useLocation, useNavigate } from 'react-router-dom';
import { Card, CardContent } from "../components/ui/card";
import { Button } from "../components/ui/button";
import { AlertTriangle, Home, RefreshCw } from 'lucide-react';
import JobStatusTracker from '../components/JobStatusTracker';

export default function JobResults() {
  const { jobId } = useParams();
  const location = useLocation();
  const navigate = useNavigate();
  
  const [jobStatus, setJobStatus] = useState('pending');
  const [error, setError] = useState(null);
  const [resultAnalysisId, setResultAnalysisId] = useState(null);
  
  const { documentTitle, analysisType, userId } = location.state || {};

  console.log('🔍 JobResults mounted:', { jobId, userId, documentTitle, analysisType });

  // Polling for job status
  useEffect(() => {
    if (!jobId || !userId) {
      console.error('❌ Missing required data:', { jobId, userId });
      setError('Missing job ID or user ID. Please try uploading again.');
      return;
    }

    let pollInterval;
    let pollCount = 0;
    const MAX_POLLS = 60; // 2 minutes max (60 * 2 seconds)

    const pollJobStatus = async () => {
      try {
        pollCount++;
        console.log(`🔄 Polling job status (${pollCount}/${MAX_POLLS}):`, jobId);
        
        const res = await fetch(
          `http://localhost:8000/job-status/${jobId}?user_id=${userId}`
        );

        if (!res.ok) {
          const errorData = await res.json();
          throw new Error(errorData.detail || `HTTP ${res.status}`);
        }

        const data = await res.json();
        console.log('📊 Job status update:', data);

        setJobStatus(data.status);

        // When completed, fetch results and navigate
        if (data.status === 'completed') {
          clearInterval(pollInterval);
          setResultAnalysisId(data.resultAnalysisId);
          console.log('✅ Job completed! Analysis ID:', data.resultAnalysisId);
          
          // Fetch full results and navigate
          if (data.resultAnalysisId) {
            await fetchAndNavigateToResults(data.resultAnalysisId);
          } else {
            setError('Analysis completed but no result ID found');
          }
        } else if (data.status === 'failed') {
          clearInterval(pollInterval);
          setError(data.errorMessage || 'Analysis failed. Please try again.');
          console.error('❌ Job failed:', data.errorMessage);
        }

        // Stop polling after max attempts
        if (pollCount >= MAX_POLLS) {
          clearInterval(pollInterval);
          setError('Analysis is taking longer than expected. Please check your dashboard later.');
          console.warn('⏱️ Max polling attempts reached');
        }
      } catch (err) {
        console.error('❌ Error polling job status:', err);
        setError(`Failed to check job status: ${err.message}`);
        clearInterval(pollInterval);
      }
    };

    // Initial poll immediately
    pollJobStatus();
    
    // Then poll every 2 seconds
    pollInterval = setInterval(pollJobStatus, 2000);

    // Cleanup on unmount
    return () => {
      if (pollInterval) {
        console.log('🧹 Cleaning up polling interval');
        clearInterval(pollInterval);
      }
    };
  }, [jobId, userId]);

  // Fetch full analysis results and navigate to results page
  const fetchAndNavigateToResults = async (analysisId) => {
    try {
      console.log(`📥 Fetching analysis results: ${analysisId}`);
      
      const res = await fetch(
        `http://localhost:8000/analysis-result/${analysisId}?user_id=${userId}`
      );

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || `HTTP ${res.status}`);
      }

      const data = await res.json();
      console.log('✅ Analysis results fetched:', data);

      // Navigate to results page with full data
      setTimeout(() => {
        navigate(`/results/${analysisId}`, {
          state: {
            analysis: data,
            analysisType: analysisType || data.analysisType,
            analysisId: analysisId
          }
        });
      }, 1500); // 1.5s delay to show completion animation
      
    } catch (err) {
      console.error('❌ Error fetching analysis results:', err);
      setError(`Failed to load results: ${err.message}`);
    }
  };

  const handleComplete = () => {
    console.log('🎉 Analysis complete callback triggered');
    // Navigation is handled in fetchAndNavigateToResults
  };

  const handleRetry = () => {
    navigate('/upload');
  };

  // Error state
  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-black via-[#0F2A40] to-[#064E3B] py-16 px-4">
        <div className="max-w-2xl mx-auto">
          <Card className="border-none bg-red-900/20 backdrop-blur-md shadow-2xl">
            <CardContent className="p-8">
              <div className="text-center space-y-4">
                <AlertTriangle className="w-16 h-16 text-red-500 mx-auto" />
                <h2 className="text-2xl font-bold text-white">Analysis Error</h2>
                <p className="text-gray-300">{error}</p>
                <div className="flex gap-4 justify-center pt-4">
                  <Button
                    onClick={handleRetry}
                    className="bg-gradient-to-r from-cyan-600 to-cyan-500 hover:from-cyan-500 hover:to-cyan-400"
                  >
                    <RefreshCw className="w-4 h-4 mr-2" />
                    Try Again
                  </Button>
                  <Button
                    onClick={() => navigate('/dashboard')}
                    variant="outline"
                    className="border-gray-500 text-white hover:bg-gray-800"
                  >
                    <Home className="w-4 h-4 mr-2" />
                    Go to Dashboard
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    );
  }

  // Main loading/processing view
  return (
    <div className="min-h-screen bg-gradient-to-b from-black via-[#0F2A40] to-[#064E3B] py-16 px-4">
      <div className="max-w-3xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">
            Document Analysis in Progress
          </h1>
          <p className="text-gray-300">
            Please wait while we analyze your document
          </p>
        </div>

        {/* Job Status Tracker */}
        <div className="mb-6">
          <JobStatusTracker
            status={jobStatus}
            documentTitle={documentTitle || 'Your Document'}
            onComplete={handleComplete}
            estimatedTime="30-60 seconds"
            errorMessage={error}
          />
        </div>

        {/* Info Card - Only show during pending/processing */}
        {(jobStatus === 'pending' || jobStatus === 'processing') && (
          <Card className="border-none bg-[#064E3B]/80 backdrop-blur-md shadow-xl">
            <CardContent className="p-6">
              <h3 className="text-white font-semibold text-lg mb-3">
                What's happening now?
              </h3>
              <ul className="space-y-2 text-gray-300 text-sm">
                <li className="flex items-start gap-2">
                  <span className="text-cyan-400">•</span>
                  <span>Your document has been securely uploaded to cloud storage</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-cyan-400">•</span>
                  <span>Personal information (PII) is being automatically redacted for privacy</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-cyan-400">•</span>
                  <span>AI is analyzing contract clauses for potential risks and unfair terms</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-cyan-400">•</span>
                  <span>Generating personalized recommendations and negotiation strategies</span>
                </li>
              </ul>
              <div className="mt-6 p-4 bg-cyan-900/30 border border-cyan-500/30 rounded-lg">
                <p className="text-cyan-200 text-sm">
                  💡 <strong>Tip:</strong> You can safely close this page and check back later. 
                  Your analysis will be saved to your dashboard automatically.
                </p>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Action Buttons */}
        <div className="mt-8 flex justify-center gap-4">
          <Button
            onClick={() => navigate('/dashboard')}
            variant="outline"
            className="border-gray-500 text-white hover:bg-gray-800"
          >
            <Home className="w-4 h-4 mr-2" />
            Go to Dashboard
          </Button>
          
          {jobStatus === 'failed' && (
            <Button
              onClick={handleRetry}
              className="bg-gradient-to-r from-cyan-600 to-cyan-500 hover:from-cyan-500 hover:to-cyan-400"
            >
              <RefreshCw className="w-4 h-4 mr-2" />
              Try Again
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}