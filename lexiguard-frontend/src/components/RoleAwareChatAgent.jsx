import React, { useState, useEffect, useRef } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { manageUserRole, saveConversationHistory, getConversationHistory } from '../services/firestoreService';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { 
  MessageSquare, 
  Send, 
  User, 
  Bot, 
  Loader2, 
  RefreshCw,
  AlertCircle,
  CheckCircle,
  UserCheck
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';

/**
 * Role-Aware Intelligent Chat Agent Component
 * 
 * Features:
 * - Role discovery and persistence
 * - Context-aware conversations
 * - Intent-based routing
 * - Conversation history management
 * - Scalable state management
 */
export default function RoleAwareChatAgent({ 
  analysisId, 
  redactedDocumentText, 
  className = "",
  showTitle = true,
  height = "600px"  // ✅ Increased from 500px to 600px for better readability
}) {
  const { currentUser } = useAuth();
  
  // Debug logging
  console.log('🚀 RoleAwareChatAgent rendered with props:', {
    analysisId,
    hasRedactedText: !!redactedDocumentText,
    redactedTextLength: redactedDocumentText?.length || 0,
    currentUserExists: !!currentUser,
    currentUserUid: currentUser?.uid,
    height
  });
  
  // Chat state
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [chatInitialized, setChatInitialized] = useState(false);
  const [initializationInProgress, setInitializationInProgress] = useState(false); // ✅ Prevent double initialization
  
  // Role state
  const [userRole, setUserRole] = useState(null);
  const [roleDiscoveryComplete, setRoleDiscoveryComplete] = useState(false);
  const [showRoleBadge, setShowRoleBadge] = useState(false);
  
  // Error and connection state
  const [connectionError, setConnectionError] = useState(null);
  
  // Document text with fallback
  const documentText = redactedDocumentText || 'No document text available';
  
  // Refs
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const typewriterIntervals = useRef(new Set()); // ✅ Track intervals for cleanup
  const hasInitialized = useRef(false); // ✅ Prevent React StrictMode double initialization
  
  // Auto-scroll to bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ 
      behavior: 'smooth',
      block: 'end'
    });
  };
  
  useEffect(() => {
    // Add a small delay to ensure DOM is updated
    const timeoutId = setTimeout(() => {
      scrollToBottom();
    }, 100);
    
    return () => clearTimeout(timeoutId);
  }, [messages]);

  // ✅ Cleanup typewriter intervals on unmount
  useEffect(() => {
    return () => {
      typewriterIntervals.current.forEach(interval => clearInterval(interval));
      typewriterIntervals.current.clear();
    };
  }, []);
  
  // Initialize chat and load previous conversation
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    console.log('🔄 useEffect triggered:', {
      chatInitialized,
      initializationInProgress,
      hasInitializedRef: hasInitialized.current,
      analysisId: !!analysisId,
      currentUser: !!currentUser,
      messagesLength: messages.length
    });
    
    // ✅ Triple protection against double initialization
    if (!chatInitialized && 
        !initializationInProgress && 
        !hasInitialized.current && 
        analysisId && 
        currentUser) {
      console.log('✅ Starting chat initialization...');
      hasInitialized.current = true; // ✅ Prevent React StrictMode double calls
      setInitializationInProgress(true);
      initializeChat();
    } else {
      console.log('❌ Chat initialization skipped:', {
        chatInitialized,
        initializationInProgress,
        hasInitializedRef: hasInitialized.current,
        hasAnalysisId: !!analysisId,
        hasCurrentUser: !!currentUser,
        messagesLength: messages.length
      });
    }
  }, [chatInitialized, analysisId, currentUser]);
  
  const initializeChat = async () => {
    try {
      setIsLoading(true);
      
      // Load existing role and conversation history
      const [existingRole, existingHistory] = await Promise.all([
        manageUserRole(analysisId, currentUser.uid),
        getConversationHistory(analysisId, currentUser.uid)
      ]);
      
      if (existingRole) {
        setUserRole(existingRole);
        setRoleDiscoveryComplete(true);
        setShowRoleBadge(true);
        console.log(`✅ Loaded existing role: ${existingRole}`);
      }
      
      if (existingHistory && existingHistory.length > 0) {
        setMessages(existingHistory);
        console.log(`✅ Loaded ${existingHistory.length} previous messages`);
      } else if (messages.length === 0) { // ✅ Only add initial message if no messages exist
        // Start with initial role discovery if no role exists
        if (!existingRole) {
          console.log('🎯 Adding role discovery message (no existing role, no existing history)');
          // ✅ Use typewriter effect for role discovery message
          addMessageWithTypewriter('ai', 
            "Hello! I'm here to help you understand this document. " +
            "To give you the most relevant and personalized insights, " +
            "**what's your role** in this document?\n\n" +
            "For example:\n" +
            "• Tenant or Landlord (for leases)\n" +
            "• Employee or Employer (for contracts)\n" +
            "• Buyer or Seller (for agreements)\n" +
            "• Borrower or Lender (for loans)\n" +
            "• Client or Service Provider\n" +
            "• Or just tell me like: \"I'm the tenant\" or \"I'm party A\""
          );
        } else {
          console.log('🎯 Adding welcome message (existing role, no existing history)');
          // If role exists but no history, show a welcome message
          addMessageWithTypewriter('ai', `Hello! I'm ready to help you understand this document from your perspective as the **${existingRole}**. What questions do you have?`);
        }
      } else {
        console.log('🔄 Skipping initial message - messages already exist:', messages.length);
      }
      
      setChatInitialized(true);
    } catch (error) {
      console.error('❌ Chat initialization error:', error);
      setConnectionError('Failed to initialize chat. Please refresh the page.');
      addMessage('ai', 'Sorry, I encountered an error during initialization. Please refresh the page and try again.');
    } finally {
      setIsLoading(false);
      setInitializationInProgress(false); // ✅ Reset initialization flag
    }
  };
  
  const handleRoleIdentification = async (role) => {
    try {
      setUserRole(role);
      setRoleDiscoveryComplete(true);
      setShowRoleBadge(true);
      
      // Save role to Firestore
      await manageUserRole(analysisId, currentUser.uid, role);
      console.log(`✅ Role identified and saved: ${role}`);
    } catch (error) {
      console.error('❌ Error saving role:', error);
    }
  };
  
  const addMessage = (sender, content, metadata = {}) => {
    const message = {
      id: Date.now() + Math.random(),
      sender,
      content: content || '', // Ensure content is never undefined/null
      timestamp: new Date(),
      isTyping: sender === 'ai', // ✅ Flag for typewriter effect
      ...metadata
    };
    
    setMessages(prev => [...prev, message]);
    return message;
  };

  // ✅ Add typewriter effect for AI messages
  const addMessageWithTypewriter = (sender, content, metadata = {}) => {
    if (sender !== 'ai' || !content) {
      return addMessage(sender, content, metadata);
    }

    // Add message with empty content first
    const message = {
      id: Date.now() + Math.random(),
      sender,
      content: '',
      timestamp: new Date(),
      isTyping: true,
      fullContent: content, // Store full content for typewriter effect
      ...metadata
    };
    
    setMessages(prev => [...prev, message]);

    // Start typewriter effect
    let currentText = '';
    let currentIndex = 0;
    const words = content.split(' ');
    
    const typeInterval = setInterval(() => {
      if (currentIndex < words.length) {
        currentText += (currentIndex > 0 ? ' ' : '') + words[currentIndex];
        currentIndex++;
        
        // Update the message content
        setMessages(prev => prev.map(msg => 
          msg.id === message.id 
            ? { ...msg, content: currentText }
            : msg
        ));
      } else {
        // Finished typing
        clearInterval(typeInterval);
        typewriterIntervals.current.delete(typeInterval); // ✅ Remove from tracking
        setMessages(prev => prev.map(msg => 
          msg.id === message.id 
            ? { ...msg, isTyping: false }
            : msg
        ));
      }
    }, 50); // Adjust speed: 50ms per word (fast but readable)

    typewriterIntervals.current.add(typeInterval); // ✅ Track interval for cleanup
    return message;
  };
  
  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;
    
    const userMessage = inputMessage.trim();
    setInputMessage('');
    
    // Add user message to UI
    addMessage('user', userMessage);
    
    try {
      const response = await fetchChatResponse(userMessage, true);
      
      // Handle role identification responses
      if (!roleDiscoveryComplete && response.identified_role) {
        await handleRoleIdentification(response.identified_role);
      }
      
      // Add AI response - ensure reply exists and use typewriter effect
      const aiReply = response.reply || 'I encountered an issue processing your request.';
      addMessageWithTypewriter('ai', aiReply, {
        intent: response.intent,
        role: response.identified_role
      });
      
      // Save conversation history
      await saveConversationToFirestore();
      
    } catch (error) {
      console.error('❌ Send message error:', error);
      addMessage('ai', 'Sorry, I encountered an error. Please try again.');
      setConnectionError('Connection error. Please check your internet connection.');
    }
  };
  
  const fetchChatResponse = async (message, includeHistory = true) => {
    setIsLoading(true);
    setConnectionError(null);
    
    try {
      // Prepare conversation history for backend (last 6 messages for context)
      const historyForBackend = includeHistory 
        ? messages.slice(-6).map(msg => ({
            sender: msg.sender,
            text: msg.content
          }))
        : [];
      
      const requestBody = {
        message: message,
        document_text: redactedDocumentText,
        analysis_id: analysisId,
        user_role: userRole,
        conversation_history: historyForBackend
      };
      
      console.log('📤 Sending chat request:', {
        message,
        hasDocumentText: !!redactedDocumentText,
        analysisId,
        userRole,
        historyLength: historyForBackend.length
      });
      
      const response = await fetch(`${BACKEND_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `HTTP ${response.status}`);
      }
      
      const data = await response.json();
      console.log('📦 Chat response:', data);
      
      return data;
    } catch (error) {
      console.error('❌ Chat API error:', error);
      throw error;
    } finally {
      setIsLoading(false);
    }
  };
  
  const saveConversationToFirestore = async () => {
    try {
      if (messages.length > 0) {
        // Convert messages to the format expected by Firestore
        const conversationHistory = messages.map(msg => ({
          sender: msg.sender,
          text: msg.content,
          timestamp: msg.timestamp
        }));
        
        await saveConversationHistory(analysisId, currentUser.uid, conversationHistory);
      }
    } catch (error) {
      console.error('❌ Error saving conversation:', error);
      // Don't show error to user for this background operation
    }
  };
  
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };
  
  const resetChat = async () => {
    try {
      setMessages([]);
      setUserRole(null);
      setRoleDiscoveryComplete(false);
      setShowRoleBadge(false);
      setChatInitialized(false);
      setInitializationInProgress(false); // ✅ Reset initialization flag
      hasInitialized.current = false; // ✅ Reset ref for fresh initialization
      setConnectionError(null);
      
      // Clear role from Firestore
      await manageUserRole(analysisId, currentUser.uid, '');
      await saveConversationHistory(analysisId, currentUser.uid, []);
      
      // Reinitialize
      setTimeout(() => {
        initializeChat();
      }, 500);
    } catch (error) {
      console.error('❌ Error resetting chat:', error);
    }
  };
  
  if (!analysisId) {
    console.log('❌ RoleAwareChatAgent: No analysisId provided');
    return (
      <Card className={`border-none bg-[#064E3B]/90 backdrop-blur-md shadow-2xl ${className}`}>
        <CardContent className="p-6 text-center">
          <AlertCircle className="w-12 h-12 text-amber-400 mx-auto mb-4" />
          <p className="text-gray-300">Chat is not available - Missing analysis ID.</p>
        </CardContent>
      </Card>
    );
  }
  
  return (
    <Card 
      className={`border-none bg-[#064E3B]/90 backdrop-blur-md shadow-2xl flex flex-col ${className}`} 
      style={{ 
        height: height, 
        maxHeight: height, 
        minHeight: '500px', // ✅ Increased minimum height for better UX
        overflow: 'hidden' // Prevent any content from expanding beyond container
      }}
    >
      {showTitle && (
        <CardHeader className="border-b border-gray-700/50 flex-shrink-0">
          <CardTitle className="flex items-center justify-between text-white text-lg">
            <div className="flex items-center gap-2">
              <MessageSquare className="w-5 h-5 text-cyan-400" />
              Chat with Document
              {/* Debug indicator */}
              <span className="text-xs text-green-400">●</span>
            </div>
            <div className="flex items-center gap-2">
              {showRoleBadge && userRole && (
                <motion.div
                  initial={{ scale: 0, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  transition={{ duration: 0.3 }}
                >
                  <Badge 
                    variant="secondary" 
                    className="bg-cyan-500/20 text-cyan-300 border-cyan-500/30 flex items-center gap-1"
                  >
                    <UserCheck className="w-3 h-3" />
                    {userRole}
                  </Badge>
                </motion.div>
              )}
              <Button
                onClick={resetChat}
                size="sm"
                variant="ghost"
                className="text-gray-400 hover:text-white h-8 w-8 p-0"
                title="Reset conversation"
              >
                <RefreshCw className="w-4 h-4" />
              </Button>
            </div>
          </CardTitle>
        </CardHeader>
      )}
      
      <CardContent className="flex flex-col flex-1 overflow-hidden p-4 min-h-0">
        {/* Connection Error Banner */}
        {connectionError && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-4 p-3 bg-red-900/30 border border-red-500/50 rounded-lg flex items-center gap-2"
          >
            <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0" />
            <span className="text-red-200 text-sm">{connectionError}</span>
          </motion.div>
        )}
        
        {/* Messages Container - Fixed height with scrolling */}
        <div
          className="flex-1 flex flex-col gap-3 overflow-y-auto overflow-x-hidden px-4 py-6 scroll-smooth"
          style={{
            scrollbarWidth: "thin",
            scrollbarColor: "#0FC6B2 #0F2A40",
            maxHeight: "100%",
            flex: "1 1 0", // Ensure proper flex behavior
            minHeight: "350px", // ✅ Set minimum height for the messages container
            backgroundColor: "rgba(255, 255, 255, 0.05)", // ✅ Subtle background to make container more visible
            borderRadius: "12px", // ✅ Rounded corners for better appearance
          }}
        >
          <AnimatePresence initial={false}>
            {messages.map((message) => (
              <motion.div
                key={message.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.3 }}
                className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'} flex-shrink-0`}
              >
                <div
                  className={`max-w-[85%] p-4 rounded-2xl shadow-lg transition-all duration-200 ${
                    message.sender === 'user'
                      ? 'bg-gradient-to-r from-cyan-600 to-cyan-500 text-white'
                      : 'bg-gray-800 text-gray-100 border border-gray-700'
                  }`}
                >
                  <div className="flex items-start gap-2">
                    {message.sender === 'user' ? (
                      <User className="w-4 h-4 flex-shrink-0 mt-0.5" />
                    ) : (
                      <Bot className="w-4 h-4 flex-shrink-0 mt-0.5 text-cyan-400" />
                    )}
                    <div className="flex-1">
                      <div 
                        className="text-sm leading-relaxed"
                        dangerouslySetInnerHTML={{
                          __html: (message.content || '').replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                        }}
                      />
                      {/* ✅ Typing indicator */}
                      {message.isTyping && (
                        <div className="inline-flex items-center ml-2">
                          <div className="flex space-x-1">
                            <div className="w-1 h-1 bg-cyan-400 rounded-full animate-pulse"></div>
                            <div className="w-1 h-1 bg-cyan-400 rounded-full animate-pulse" style={{ animationDelay: '0.2s' }}></div>
                            <div className="w-1 h-1 bg-cyan-400 rounded-full animate-pulse" style={{ animationDelay: '0.4s' }}></div>
                          </div>
                        </div>
                      )}
                      {message.intent && (
                        <div className="mt-2 text-xs opacity-75">
                          Intent: {message.intent}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
          
          {/* Loading indicator */}
          {isLoading && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex justify-start"
            >
              <div className="bg-gray-800 border border-gray-700 p-4 rounded-2xl flex items-center gap-2">
                <Bot className="w-4 h-4 text-cyan-400" />
                <Loader2 className="w-4 h-4 animate-spin text-cyan-400" />
                <span className="text-gray-300 text-sm">LexiGuard is thinking...</span>
              </div>
            </motion.div>
          )}
          
          <div ref={messagesEndRef} />
        </div>
        
        {/* Input Area */}
        <div className="mt-4 flex gap-2 flex-shrink-0">
          <div className="flex-1 relative">
            <textarea
              ref={inputRef}
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyDown={handleKeyPress}
              placeholder={
                roleDiscoveryComplete 
                  ? "Ask me anything about this document..." 
                  : "Please tell me your role in this document..."
              }
              className="w-full p-3 pr-12 rounded-2xl bg-[#064E3B]/80 text-white border border-gray-600 focus:outline-none focus:border-cyan-400 placeholder-gray-400 resize-none min-h-[48px] max-h-[120px]"
              rows={1}
              disabled={isLoading}
              style={{
                scrollbarWidth: "thin",
                scrollbarColor: "#0FC6B2 #0F2A40",
              }}
            />
          </div>
          <Button
            onClick={handleSendMessage}
            disabled={isLoading || !inputMessage.trim()}
            className="bg-cyan-600 hover:bg-cyan-500 text-white rounded-2xl px-6 h-12 flex-shrink-0 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Send className="w-4 h-4" />
            )}
          </Button>
        </div>
        
        {/* Status indicator */}
        {!isLoading && roleDiscoveryComplete && (
          <div className="mt-2 flex items-center justify-center">
            <div className="flex items-center gap-2 text-xs text-gray-400">
              <CheckCircle className="w-3 h-3 text-green-400" />
              <span>Connected • Role: {userRole}</span>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}