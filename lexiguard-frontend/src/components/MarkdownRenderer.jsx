import React from 'react';

// Simple Markdown to JSX converter for summary text
const MarkdownRenderer = ({ text }) => {
  if (!text || typeof text !== 'string') return null;

  // Split by double newlines for paragraphs
  const paragraphs = text.split('\n\n');

  const renderLine = (line, index) => {
    // Handle bold text (*text* or _text_)
    line = line.replace(/\\(.?)\\*/g, '<strong>$1</strong>');
    line = line.replace(/(.*?)/g, '<strong>$1</strong>');
    
    // Handle italic text (text or text)
    line = line.replace(/\(.?)\*/g, '<em>$1</em>');
    line = line.replace(/(.*?)/g, '<em>$1</em>');
    
    // Handle headings (### text or ## text or # text)
    if (line.startsWith('### ')) {
      return <h3 key={index} className="text-lg font-bold text-white mt-4 mb-2" dangerouslySetInnerHTML={{ __html: line.replace('### ', '') }} />;
    }
    if (line.startsWith('## ')) {
      return <h2 key={index} className="text-xl font-bold text-white mt-4 mb-2" dangerouslySetInnerHTML={{ __html: line.replace('## ', '') }} />;
    }
    if (line.startsWith('# ')) {
      return <h1 key={index} className="text-2xl font-bold text-white mt-4 mb-2" dangerouslySetInnerHTML={{ __html: line.replace('# ', '') }} />;
    }
    
    // Handle bullet points (- item or * item)
    if (line.trim().startsWith('- ') || line.trim().startsWith('* ')) {
      return (
        <li key={index} className="ml-4 text-gray-200" dangerouslySetInnerHTML={{ 
          __html: line.trim().replace(/^[-*]\s/, '') 
        }} />
      );
    }
    
    // Handle numbered lists (1. item, 2. item, etc.)
    const numberedMatch = line.trim().match(/^\d+\.\s/);
    if (numberedMatch) {
      return (
        <li key={index} className="ml-4 text-gray-200" dangerouslySetInnerHTML={{ 
          __html: line.trim().replace(/^\d+\.\s/, '') 
        }} />
      );
    }
    
    // Regular paragraph
    if (line.trim()) {
      return <p key={index} className="text-gray-200 mb-2" dangerouslySetInnerHTML={{ __html: line }} />;
    }
    
    return null;
  };

  return (
    <div className="prose prose-invert max-w-none">
      {paragraphs.map((paragraph, pIndex) => {
        const lines = paragraph.split('\n');
        
        // Check if this is a list block
        const isBulletList = lines.some(l => l.trim().startsWith('- ') || l.trim().startsWith('* '));
        const isNumberedList = lines.some(l => /^\d+\.\s/.test(l.trim()));
        
        if (isBulletList) {
          return (
            <ul key={pIndex} className="list-disc ml-6 mb-4 space-y-1">
              {lines.map((line, lIndex) => renderLine(line, `${pIndex}-${lIndex}`))}
            </ul>
          );
        }
        
        if (isNumberedList) {
          return (
            <ol key={pIndex} className="list-decimal ml-6 mb-4 space-y-1">
              {lines.map((line, lIndex) => renderLine(line, `${pIndex}-${lIndex}`))}
            </ol>
          );
        }
        
        // Regular paragraph block
        return (
          <div key={pIndex} className="mb-4">
            {lines.map((line, lIndex) => renderLine(line, `${pIndex}-${lIndex}`))}
          </div>
        );
      })}
    </div>
  );
};

// Example usage
export default function MarkdownSummaryDemo() {
  const exampleSummary = `# Contract Analysis Summary

*Overview:* This employment agreement establishes the terms between the employer and employee.

## Key Points:

- *Duration:* 2-year fixed term contract starting January 1, 2024
- *Compensation:* $85,000 annual salary with quarterly reviews
- *Benefits:* Health insurance, 401(k) matching, 15 days PTO

### Important Clauses:

1. Non-compete clause extends 12 months post-termination
2. Confidentiality obligations continue indefinitely
3. Termination requires 30-day written notice

*Recommendation:* Review the non-compete terms carefully before signing.`;

  return (
    <div className="min-h-screen bg-gradient-to-b from-black via-[#0F2A40] to-[#064E3B] p-8">
      <div className="max-w-4xl mx-auto">
        <div className="bg-[#064E3B]/90 backdrop-blur-md rounded-xl shadow-2xl p-6 border border-cyan-400/20">
          <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-2">
            <svg className="w-6 h-6 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            Document Summary
          </h2>
          <MarkdownRenderer text={exampleSummary} />
        </div>
        
        <div className="mt-6 bg-gray-800/50 rounded-lg p-4 border border-gray-700">
          <p className="text-gray-300 text-sm">
            ✅ <strong>Formatted correctly:</strong> Headings, bold text, bullet points, and numbered lists are now properly rendered instead of showing raw Markdown symbols.
          </p>
        </div>
      </div>
    </div>
  );
}