// pages/Login.jsx
import React, { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { Button } from '../components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Shield, AlertCircle } from 'lucide-react';
import { motion } from 'framer-motion';

export default function Login() {
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const { signInWithGoogle } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  // Get the page user tried to access before login
  const from = location.state?.from || '/upload';

  const handleGoogleSignIn = async () => {
    try {
      setError('');
      setLoading(true);
      await signInWithGoogle();
      // Redirect to the page they tried to access or upload page
      navigate(from, { replace: true });
    } catch (error) {
      console.error('Failed to sign in:', error);
      setError('Failed to sign in with Google. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen relative bg-gradient-to-b from-black via-[#0F2A40] to-[#064E3B] overflow-hidden flex items-center justify-center py-16">
      {/* Aurora Glow Background */}
      <div className="absolute inset-0 aurora-bg opacity-20" />

      <div className="relative max-w-md w-full mx-auto px-4">
        {/* Logo and Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-8"
        >
          <div className="flex justify-center mb-4">
            <div className="w-16 h-16 bg-gradient-to-r from-[#1C3D5A] to-[#14B8A6] rounded-2xl flex items-center justify-center">
              <Shield className="w-10 h-10 text-white" />
            </div>
          </div>
          <h1 className="text-4xl font-bold text-white mb-2">Welcome to LexiGuard</h1>
          <p className="text-gray-200/70 text-lg">
            Sign in to analyze your legal documents
          </p>
        </motion.div>

        {/* Login Card */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          <Card className="border-none bg-[#064E3B]/90 backdrop-blur-md shadow-2xl">
            <CardHeader className="pb-6">
              <CardTitle className="text-center text-2xl text-white">
                Sign In to Continue
              </CardTitle>
            </CardHeader>
            <CardContent className="px-8 pb-8">
              {error && (
                <div className="mb-6 p-4 bg-red-500/20 border border-red-500/50 rounded-lg flex items-start gap-3">
                  <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
                  <p className="text-red-200 text-sm">{error}</p>
                </div>
              )}

              <div className="w-full flex justify-center mb-8">
                <button
                  onClick={handleGoogleSignIn}
                  disabled={loading}
                  className="bg-white hover:bg-gray-100 text-gray-900 font-semibold py-6 px-12 text-lg shadow-lg transition-all rounded-md disabled:opacity-50 disabled:cursor-not-allowed"
                  style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '12px', whiteSpace: 'nowrap' }}
                >
                  {loading ? (
                    <>
                      <svg
                        className="animate-spin h-5 w-5"
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
                      Signing in...
                    </>
                  ) : (
                    <>
                      <svg className="w-6 h-6" viewBox="0 0 24 24">
                        <path
                          fill="#4285F4"
                          d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                        />
                        <path
                          fill="#34A853"
                          d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                        />
                        <path
                          fill="#FBBC05"
                          d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
                        />
                        <path
                          fill="#EA4335"
                          d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                        />
                      </svg>
                      Continue with Google
                    </>
                  )}
                </button>
              </div>    

              <div className="text-center">
                <p className="text-gray-300 text-sm leading-relaxed">
                  By continuing, you agree to our{' '}
                  <a href="/terms" className="text-cyan-400 hover:underline">
                    Terms of Service
                  </a>{' '}
                  and{' '}
                  <a href="/privacy" className="text-cyan-400 hover:underline">
                    Privacy Policy
                  </a>
                </p>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Features Preview */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
          className="mt-8 text-center"
        >
          <p className="text-gray-300 text-sm mb-4">After signing in, you'll get access to:</p>
          <div className="grid grid-cols-1 gap-3">
            <div className="bg-white/10 backdrop-blur-sm rounded-lg p-3 text-left">
              <p className="text-white text-sm">✓ AI-powered document analysis</p>
            </div>
            <div className="bg-white/10 backdrop-blur-sm rounded-lg p-3 text-left">
              <p className="text-white text-sm">✓ Risk identification & recommendations</p>
            </div>
            <div className="bg-white/10 backdrop-blur-sm rounded-lg p-3 text-left">
              <p className="text-white text-sm">✓ Document history & saved analyses</p>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
}