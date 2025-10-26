import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Loader2, CheckCircle2, XCircle, Clock, FileText, Sparkles, Upload, Shield } from 'lucide-react';

/**
 * JobStatusTracker Component
 * 
 * Displays real-time status of async document analysis job
 * Shows loading animations and progress indicators
 * 
 * Props:
 * - status: 'pending' | 'processing' | 'completed' | 'failed'
 * - documentTitle: string
 * - onComplete: callback when job completes
 * - estimatedTime: string (e.g., "30-60 seconds")
 * - errorMessage: error message if failed
 */
const JobStatusTracker = ({ 
  status, 
  documentTitle, 
  onComplete,
  estimatedTime = "30-60 seconds",
  errorMessage = null
}) => {
  const [elapsedTime, setElapsedTime] = useState(0);

  // Timer for elapsed time
  useEffect(() => {
    if (status === 'pending' || status === 'processing') {
      const interval = setInterval(() => {
        setElapsedTime(prev => prev + 1);
      }, 1000);

      return () => clearInterval(interval);
    }
  }, [status]);

  // Call onComplete when job finishes
  useEffect(() => {
    if (status === 'completed' && onComplete) {
      onComplete();
    }
  }, [status, onComplete]);

  const formatTime = (seconds) => {
    if (seconds < 60) return `${seconds}s`;
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}m ${secs}s`;
  };

  const getStatusConfig = () => {
    switch (status) {
      case 'pending':
        return {
          icon: Clock,
          color: 'text-blue-400',
          bgColor: 'bg-blue-900/30',
          borderColor: 'border-blue-500/50',
          title: 'Upload Complete',
          message: 'Your document is queued for analysis...',
          showLoader: true,
          progress: 25
        };
      case 'processing':
        return {
          icon: Sparkles,
          color: 'text-purple-400',
          bgColor: 'bg-purple-900/30',
          borderColor: 'border-purple-500/50',
          title: 'Analyzing Document',
          message: 'AI is analyzing your document for risks and insights...',
          showLoader: true,
          progress: 60
        };
      case 'completed':
        return {
          icon: CheckCircle2,
          color: 'text-green-400',
          bgColor: 'bg-green-900/30',
          borderColor: 'border-green-500/50',
          title: 'Analysis Complete!',
          message: 'Your document has been analyzed successfully.',
          showLoader: false,
          progress: 100
        };
      case 'failed':
        return {
          icon: XCircle,
          color: 'text-red-400',
          bgColor: 'bg-red-900/30',
          borderColor: 'border-red-500/50',
          title: 'Analysis Failed',
          message: errorMessage || 'Something went wrong. Please try again.',
          showLoader: false,
          progress: 0
        };
      default:
        return {
          icon: FileText,
          color: 'text-gray-400',
          bgColor: 'bg-gray-900/30',
          borderColor: 'border-gray-500/50',
          title: 'Unknown Status',
          message: 'Processing...',
          showLoader: true,
          progress: 10
        };
    }
  };

  const config = getStatusConfig();
  const Icon = config.icon;

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={status}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -20 }}
        transition={{ duration: 0.3 }}
        className={`rounded-xl border-2 ${config.borderColor} ${config.bgColor} backdrop-blur-md p-6 shadow-lg`}
      >
        <div className="flex items-start space-x-4">
          {/* Status Icon */}
          <div className={`flex-shrink-0 ${config.color}`}>
            {config.showLoader ? (
              <Loader2 className="h-8 w-8 animate-spin" />
            ) : (
              <Icon className="h-8 w-8" />
            )}
          </div>

          {/* Status Content */}
          <div className="flex-1 space-y-3">
            <div>
              <h3 className={`text-xl font-bold ${config.color}`}>
                {config.title}
              </h3>
              <p className="text-sm text-gray-300 mt-1">
                {config.message}
              </p>
            </div>

            {/* Document Info */}
            <div className="bg-gray-800/50 rounded-lg p-3 border border-gray-700">
              <div className="flex items-center space-x-2">
                <FileText className="h-4 w-4 text-gray-400" />
                <span className="text-sm font-medium text-white truncate">
                  {documentTitle}
                </span>
              </div>
            </div>

            {/* Progress Indicators */}
            {(status === 'pending' || status === 'processing') && (
              <div className="space-y-3">
                {/* Progress Bar */}
                <div className="relative h-2.5 bg-gray-700 rounded-full overflow-hidden">
                  <motion.div
                    className={`absolute top-0 left-0 h-full ${
                      status === 'processing' 
                        ? 'bg-gradient-to-r from-purple-500 to-purple-400' 
                        : 'bg-gradient-to-r from-blue-500 to-blue-400'
                    }`}
                    initial={{ width: '0%' }}
                    animate={{ width: `${config.progress}%` }}
                    transition={{ duration: 1.5, ease: 'easeInOut' }}
                  />
                </div>

                {/* Time Info */}
                <div className="flex justify-between text-xs text-gray-400">
                  <span>Elapsed: {formatTime(elapsedTime)}</span>
                  <span>Est. Time: {estimatedTime}</span>
                </div>

                {/* Processing Steps */}
                <div className="space-y-2 text-xs">
                  {/* Step 1: Upload Complete */}
                  <motion.div
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    className="flex items-center space-x-2"
                  >
                    <div className="flex-shrink-0">
                      <CheckCircle2 className="h-4 w-4 text-green-400" />
                    </div>
                    <span className="text-green-300 font-medium">
                      Document uploaded to secure storage
                    </span>
                  </motion.div>

                  {/* Step 2: Queued (pending state) */}
                  {status === 'pending' && (
                    <motion.div
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: 0.2 }}
                      className="flex items-center space-x-2"
                    >
                      <div className="flex-shrink-0">
                        <Clock className="h-4 w-4 text-blue-400 animate-pulse" />
                      </div>
                      <span className="text-gray-300">
                        Job queued - waiting for worker...
                      </span>
                    </motion.div>
                  )}
                  
                  {/* Step 3: PII Redaction (processing state) */}
                  {status === 'processing' && (
                    <>
                      <motion.div
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.2 }}
                        className="flex items-center space-x-2"
                      >
                        <div className="flex-shrink-0">
                          <CheckCircle2 className="h-4 w-4 text-green-400" />
                        </div>
                        <span className="text-green-300 font-medium">
                          Worker picked up job
                        </span>
                      </motion.div>

                      <motion.div
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.4 }}
                        className="flex items-center space-x-2"
                      >
                        <div className="flex-shrink-0">
                          <Loader2 className="h-4 w-4 text-purple-400 animate-spin" />
                        </div>
                        <span className="text-purple-300">
                          Redacting personal information (PII)
                        </span>
                      </motion.div>
                      
                      <motion.div
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.6 }}
                        className="flex items-center space-x-2"
                      >
                        <div className="flex-shrink-0">
                          <Loader2 className="h-4 w-4 text-purple-400 animate-spin" />
                        </div>
                        <span className="text-purple-300">
                          AI analyzing contract clauses and risks
                        </span>
                      </motion.div>

                      <motion.div
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.8 }}
                        className="flex items-center space-x-2"
                      >
                        <div className="flex-shrink-0">
                          <Loader2 className="h-4 w-4 text-purple-400 animate-spin" />
                        </div>
                        <span className="text-purple-300">
                          Generating recommendations
                        </span>
                      </motion.div>
                    </>
                  )}
                </div>

                {/* Privacy Badge */}
                <div className="flex items-center space-x-2 pt-2 border-t border-gray-700">
                  <Shield className="h-3 w-3 text-cyan-400" />
                  <span className="text-xs text-cyan-300">
                    Your data is encrypted and will be automatically deleted after analysis
                  </span>
                </div>
              </div>
            )}

            {/* Completion Message */}
            {status === 'completed' && (
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                className="bg-green-900/40 border border-green-500/50 rounded-lg p-4"
              >
                <div className="flex items-start space-x-3">
                  <CheckCircle2 className="h-5 w-5 text-green-400 flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="text-sm text-green-300 font-semibold">
                      Analysis completed in {formatTime(elapsedTime)}
                    </p>
                    <p className="text-xs text-green-400 mt-1">
                      Redirecting to results page...
                    </p>
                  </div>
                </div>
              </motion.div>
            )}

            {/* Error Message */}
            {status === 'failed' && (
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                className="bg-red-900/40 border border-red-500/50 rounded-lg p-4"
              >
                <div className="flex items-start space-x-3">
                  <XCircle className="h-5 w-5 text-red-400 flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="text-sm text-red-300 font-semibold">
                      Analysis Failed
                    </p>
                    {errorMessage && (
                      <p className="text-xs text-red-400 mt-1">
                        {errorMessage}
                      </p>
                    )}
                    <p className="text-xs text-gray-400 mt-2">
                      Please try uploading your document again or contact support if the issue persists.
                    </p>
                  </div>
                </div>
              </motion.div>
            )}
          </div>
        </div>
      </motion.div>
    </AnimatePresence>
  );
};

export default JobStatusTracker;