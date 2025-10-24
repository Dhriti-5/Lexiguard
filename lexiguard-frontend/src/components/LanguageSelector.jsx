// components/LanguageSelector.jsx
import React, { useState, useEffect } from 'react';
import { Globe, ChevronDown, Check, Loader2, Search, X } from 'lucide-react';
import { Button } from './ui/button';
import { motion, AnimatePresence } from 'framer-motion';

// All Indian Regional Languages + International Languages
const LANGUAGES = [
  // English
  { code: 'en', name: 'English', nativeName: 'English', flag: '🇬🇧', category: 'International' },
  
  // North Indian Languages
  { code: 'hi', name: 'Hindi', nativeName: 'हिंदी', flag: '🇮🇳', category: 'North Indian' },
  { code: 'pa', name: 'Punjabi', nativeName: 'ਪੰਜਾਬੀ', flag: '🇮🇳', category: 'North Indian' },
  { code: 'ur', name: 'Urdu', nativeName: 'اردو', flag: '🇮🇳', category: 'North Indian' },
  { code: 'ks', name: 'Kashmiri', nativeName: 'کٲشُر', flag: '🇮🇳', category: 'North Indian' },
  { code: 'sd', name: 'Sindhi', nativeName: 'سنڌي', flag: '🇮🇳', category: 'North Indian' },
  { code: 'ne', name: 'Nepali', nativeName: 'नेपाली', flag: '🇮🇳', category: 'North Indian' },
  { code: 'doi', name: 'Dogri', nativeName: 'डोगरी', flag: '🇮🇳', category: 'North Indian' },
  
  // East Indian Languages
  { code: 'bn', name: 'Bengali', nativeName: 'বাংলা', flag: '🇮🇳', category: 'East Indian' },
  { code: 'as', name: 'Assamese', nativeName: 'অসমীয়া', flag: '🇮🇳', category: 'East Indian' },
  { code: 'or', name: 'Odia', nativeName: 'ଓଡ଼ିଆ', flag: '🇮🇳', category: 'East Indian' },
  { code: 'mni', name: 'Manipuri', nativeName: 'মৈতৈলোন্', flag: '🇮🇳', category: 'East Indian' },
  { code: 'sat', name: 'Santali', nativeName: 'ᱥᱟᱱᱛᱟᱲᱤ', flag: '🇮🇳', category: 'East Indian' },
  { code: 'mai', name: 'Maithili', nativeName: 'मैथिली', flag: '🇮🇳', category: 'East Indian' },
  
  // West Indian Languages
  { code: 'gu', name: 'Gujarati', nativeName: 'ગુજરાતી', flag: '🇮🇳', category: 'West Indian' },
  { code: 'mr', name: 'Marathi', nativeName: 'मराठी', flag: '🇮🇳', category: 'West Indian' },
  { code: 'kok', name: 'Konkani', nativeName: 'कोंकणी', flag: '🇮🇳', category: 'West Indian' },
  
  // South Indian Languages
  { code: 'te', name: 'Telugu', nativeName: 'తెలుగు', flag: '🇮🇳', category: 'South Indian' },
  { code: 'ta', name: 'Tamil', nativeName: 'தமிழ்', flag: '🇮🇳', category: 'South Indian' },
  { code: 'kn', name: 'Kannada', nativeName: 'ಕನ್ನಡ', flag: '🇮🇳', category: 'South Indian' },
  { code: 'ml', name: 'Malayalam', nativeName: 'മലയാളം', flag: '🇮🇳', category: 'South Indian' },
  
  // Classical Languages
  { code: 'sa', name: 'Sanskrit', nativeName: 'संस्कृत', flag: '🇮🇳', category: 'Classical' },
  
  // International Languages
  { code: 'es', name: 'Spanish', nativeName: 'Español', flag: '🇪🇸', category: 'International' },
  { code: 'fr', name: 'French', nativeName: 'Français', flag: '🇫🇷', category: 'International' },
  { code: 'de', name: 'German', nativeName: 'Deutsch', flag: '🇩🇪', category: 'International' },
  { code: 'pt', name: 'Portuguese', nativeName: 'Português', flag: '🇵🇹', category: 'International' },
  { code: 'ru', name: 'Russian', nativeName: 'Русский', flag: '🇷🇺', category: 'International' },
  { code: 'ja', name: 'Japanese', nativeName: '日本語', flag: '🇯🇵', category: 'International' },
  { code: 'zh-CN', name: 'Chinese', nativeName: '中文', flag: '🇨🇳', category: 'International' },
  { code: 'ar', name: 'Arabic', nativeName: 'العربية', flag: '🇸🇦', category: 'International' },
  { code: 'ko', name: 'Korean', nativeName: '한국어', flag: '🇰🇷', category: 'International' },
  { code: 'it', name: 'Italian', nativeName: 'Italiano', flag: '🇮🇹', category: 'International' },
];

// Group languages by category
const LANGUAGE_CATEGORIES = [
  'North Indian',
  'East Indian',
  'West Indian',
  'South Indian',
  'Classical',
  'International'
];

export default function LanguageSelector({ 
  selectedLanguage, 
  onLanguageChange, 
  loading = false,
  className = '' 
}) {
  const [isOpen, setIsOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [currentLanguage, setCurrentLanguage] = useState(
    LANGUAGES.find(lang => lang.code === selectedLanguage) || LANGUAGES[0]
  );

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (isOpen && !event.target.closest('.language-selector')) {
        setIsOpen(false);
        setSearchTerm('');
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isOpen]);

  const handleLanguageSelect = (language) => {
    setCurrentLanguage(language);
    setIsOpen(false);
    setSearchTerm('');
    onLanguageChange(language.code);
  };

  // Filter languages based on search
  const filteredLanguages = LANGUAGES.filter(lang => 
    lang.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    lang.nativeName.toLowerCase().includes(searchTerm.toLowerCase()) ||
    lang.category.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // Group filtered languages by category
  const groupedLanguages = LANGUAGE_CATEGORIES.reduce((acc, category) => {
    const langsInCategory = filteredLanguages.filter(lang => lang.category === category);
    if (langsInCategory.length > 0) {
      acc[category] = langsInCategory;
    }
    return acc;
  }, {});

  return (
    <div className={`relative language-selector ${className}`}>
      <Button
        onClick={() => !loading && setIsOpen(!isOpen)}
        disabled={loading}
        className="flex items-center gap-2 bg-[#064E3B]/90 hover:bg-[#064E3B] text-white border border-cyan-400/30 hover:border-cyan-400/50 transition-all"
      >
        {loading ? (
          <Loader2 className="w-4 h-4 animate-spin" />
        ) : (
          <Globe className="w-4 h-4" />
        )}
        <span className="font-medium hidden sm:inline">
          {currentLanguage.flag} {currentLanguage.name}
        </span>
        <span className="font-medium sm:hidden">
          {currentLanguage.flag}
        </span>
        <ChevronDown className={`w-4 h-4 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </Button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: -10, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -10, scale: 0.95 }}
            transition={{ duration: 0.2 }}
            className="absolute top-full mt-2 right-0 w-96 max-w-[calc(100vw-2rem)] bg-gray-800 border border-gray-700 rounded-lg shadow-2xl overflow-hidden z-50"
          >
            {/* Header with Search */}
            <div className="p-3 border-b border-gray-700 bg-gray-900/50">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                <input
                  type="text"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  placeholder="Search languages..."
                  className="w-full pl-10 pr-10 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-cyan-500 text-sm"
                  autoFocus
                />
                {searchTerm && (
                  <button
                    onClick={() => setSearchTerm('')}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-white"
                  >
                    <X className="w-4 h-4" />
                  </button>
                )}
              </div>
            </div>

            {/* Language List */}
            <div className="max-h-96 overflow-y-auto" style={{ scrollbarWidth: 'thin', scrollbarColor: '#4B5563 #1F2937' }}>
              {Object.keys(groupedLanguages).length === 0 ? (
                <div className="p-8 text-center text-gray-400">
                  <Globe className="w-12 h-12 mx-auto mb-3 opacity-50" />
                  <p>No languages found</p>
                </div>
              ) : (
                Object.entries(groupedLanguages).map(([category, languages]) => (
                  <div key={category} className="py-2">
                    <div className="px-3 py-2 text-xs text-cyan-400 font-semibold uppercase tracking-wide bg-gray-900/50">
                      {category} Languages
                    </div>
                    <div className="px-2">
                      {languages.map((language) => (
                        <button
                          key={language.code}
                          onClick={() => handleLanguageSelect(language)}
                          className={`w-full text-left px-3 py-2.5 rounded-md transition-colors flex items-center justify-between group ${
                            currentLanguage.code === language.code
                              ? 'bg-cyan-600 text-white'
                              : 'hover:bg-gray-700 text-gray-200'
                          }`}
                        >
                          <div className="flex items-center gap-3 flex-1 min-w-0">
                            <span className="text-xl flex-shrink-0">{language.flag}</span>
                            <div className="flex-1 min-w-0">
                              <div className="font-medium truncate">{language.name}</div>
                              <div className={`text-xs truncate ${
                                currentLanguage.code === language.code ? 'text-cyan-100' : 'text-gray-400'
                              }`}>
                                {language.nativeName}
                              </div>
                            </div>
                          </div>
                          {currentLanguage.code === language.code && (
                            <Check className="w-4 h-4 text-white flex-shrink-0 ml-2" />
                          )}
                        </button>
                      ))}
                    </div>
                  </div>
                ))
              )}
            </div>
            
            {/* Footer */}
            <div className="px-4 py-3 bg-gray-900/50 border-t border-gray-700">
              <p className="text-xs text-gray-400 text-center">
                Translation powered by Google Cloud • {LANGUAGES.length} languages supported
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}