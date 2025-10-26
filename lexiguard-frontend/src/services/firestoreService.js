// services/firestoreService.js
import { 
  collection, 
  addDoc, 
  getDocs, 
  getDoc,
  doc, 
  updateDoc, 
  deleteDoc, 
  query, 
  where, 
  orderBy,
  serverTimestamp 
} from 'firebase/firestore';
import { db } from '../firebase/config';
import { onSnapshot } from 'firebase/firestore';

const COLLECTION_NAME = 'userAnalyses';

/**
 * Save analysis to Firestore
 * @param {string} userId - Firebase Auth user ID
 * @param {object} analysisData - Analysis data from backend
 * @returns {Promise<string>} - Document ID
 */
export const saveAnalysis = async (userId, analysisData) => {
  try {
    const analysisDoc = {
      userID: userId,
      documentTitle: analysisData.documentTitle || `Analysis of ${analysisData.filename || 'Document'}`,
      originalFilename: analysisData.filename || 'Unknown',
      uploadTimestamp: serverTimestamp(),
      fileType: analysisData.file_type || analysisData.fileType || 'Text',
      piiRedacted: analysisData.pii_redacted || analysisData.piiRedacted || false,
      
      // Store redacted document text (the core privacy-preserving content)
      redactedDocumentText: analysisData.redacted_text || 
                           analysisData.redacted_document_text || 
                           analysisData.redactedDocumentText || '',
      
      // Store analysis results
      summary: analysisData.summary || '',
      
      // Handle both standard and detailed analysis formats
      risks: analysisData.risks || [],
      clauses: analysisData.clauses || [],
      
      // Analysis type for proper rendering
      analysisType: analysisData.clauses && Array.isArray(analysisData.clauses) && analysisData.clauses.length > 0 
        ? 'detailed' 
        : 'standard',
      
      // Optional fields
      total_risky_clauses: analysisData.total_risky_clauses || 
                          (analysisData.clauses ? analysisData.clauses.length : 0),
      suggestions: analysisData.suggestions || [],
      fairness_analysis: analysisData.fairness_analysis || [],
      
      // User organization features
      tags: analysisData.tags || [],
      starred: false,
      notes: '',
    };

    const docRef = await addDoc(collection(db, COLLECTION_NAME), analysisDoc);
    console.log('✅ Analysis saved to Firestore:', docRef.id);
    return docRef.id;
  } catch (error) {
    console.error('❌ Error saving analysis to Firestore:', error);
    throw error;
  }
};

/**
 * Get all analyses for a user
 * @param {string} userId - Firebase Auth user ID
 * @returns {Promise<Array>} - Array of analysis documents with IDs
 */
export const getUserAnalyses = async (userId) => {
  try {
    const q = query(
      collection(db, COLLECTION_NAME),
      where('userID', '==', userId),
      orderBy('uploadTimestamp', 'desc')
    );
    
    const querySnapshot = await getDocs(q);
    const analyses = [];
    
    querySnapshot.forEach((doc) => {
      analyses.push({
        id: doc.id,
        ...doc.data(),
        uploadTimestamp: doc.data().uploadTimestamp?.toDate() || new Date()
      });
    });
    
    console.log(`✅ Retrieved ${analyses.length} analyses for user`);
    return analyses;
  } catch (error) {
    console.error('❌ Error fetching user analyses:', error);
    throw error;
  }
};

/**
 * Get a single analysis by ID
 * @param {string} analysisId - Document ID
 * @param {string} userId - Firebase Auth user ID (for security)
 * @returns {Promise<object>} - Analysis document
 */
export const getAnalysisById = async (analysisId, userId) => {
  try {
    const docRef = doc(db, COLLECTION_NAME, analysisId);
    const docSnap = await getDoc(docRef);
    
    if (!docSnap.exists()) {
      throw new Error('Analysis not found');
    }
    
    const data = docSnap.data();
    
    // Security check: ensure this analysis belongs to the user
    if (data.userID !== userId) {
      throw new Error('Unauthorized access to analysis');
    }
    
    return {
      id: docSnap.id,
      ...data,
      uploadTimestamp: data.uploadTimestamp?.toDate() || new Date()
    };
  } catch (error) {
    console.error('❌ Error fetching analysis:', error);
    throw error;
  }
};

/**
 * Update analysis document title
 * @param {string} analysisId - Document ID
 * @param {string} userId - Firebase Auth user ID (for security)
 * @param {string} newTitle - New document title
 */
export const updateAnalysisTitle = async (analysisId, userId, newTitle) => {
  try {
    const docRef = doc(db, COLLECTION_NAME, analysisId);
    const docSnap = await getDoc(docRef);
    
    if (!docSnap.exists()) {
      throw new Error('Analysis not found');
    }
    
    // Security check
    if (docSnap.data().userID !== userId) {
      throw new Error('Unauthorized');
    }
    
    await updateDoc(docRef, {
      documentTitle: newTitle
    });
    
    console.log('✅ Analysis title updated');
  } catch (error) {
    console.error('❌ Error updating analysis title:', error);
    throw error;
  }
};

/**
 * Toggle starred status
 * @param {string} analysisId - Document ID
 * @param {string} userId - Firebase Auth user ID
 * @param {boolean} starred - New starred status
 */
export const toggleStarredAnalysis = async (analysisId, userId, starred) => {
  try {
    const docRef = doc(db, COLLECTION_NAME, analysisId);
    const docSnap = await getDoc(docRef);
    
    if (!docSnap.exists()) {
      throw new Error('Analysis not found');
    }
    
    // Security check
    if (docSnap.data().userID !== userId) {
      throw new Error('Unauthorized');
    }
    
    await updateDoc(docRef, {
      starred: starred
    });
    
    console.log('✅ Analysis starred status updated');
  } catch (error) {
    console.error('❌ Error updating starred status:', error);
    throw error;
  }
};

/**
 * Add tags to analysis
 * @param {string} analysisId - Document ID
 * @param {string} userId - Firebase Auth user ID
 * @param {Array<string>} tags - Array of tag strings
 */
export const updateAnalysisTags = async (analysisId, userId, tags) => {
  try {
    const docRef = doc(db, COLLECTION_NAME, analysisId);
    const docSnap = await getDoc(docRef);
    
    if (!docSnap.exists()) {
      throw new Error('Analysis not found');
    }
    
    // Security check
    if (docSnap.data().userID !== userId) {
      throw new Error('Unauthorized');
    }
    
    await updateDoc(docRef, {
      tags: tags
    });
    
    console.log('✅ Analysis tags updated');
  } catch (error) {
    console.error('❌ Error updating tags:', error);
    throw error;
  }
};

/**
 * Add notes to analysis
 * @param {string} analysisId - Document ID
 * @param {string} userId - Firebase Auth user ID
 * @param {string} notes - User notes
 */
export const updateAnalysisNotes = async (analysisId, userId, notes) => {
  try {
    const docRef = doc(db, COLLECTION_NAME, analysisId);
    const docSnap = await getDoc(docRef);
    
    if (!docSnap.exists()) {
      throw new Error('Analysis not found');
    }
    
    // Security check
    if (docSnap.data().userID !== userId) {
      throw new Error('Unauthorized');
    }
    
    await updateDoc(docRef, {
      notes: notes
    });
    
    console.log('✅ Analysis notes updated');
  } catch (error) {
    console.error('❌ Error updating notes:', error);
    throw error;
  }
};

/**
 * Delete an analysis
 * @param {string} analysisId - Document ID
 * @param {string} userId - Firebase Auth user ID (for security)
 */
export const deleteAnalysis = async (analysisId, userId) => {
  try {
    const docRef = doc(db, COLLECTION_NAME, analysisId);
    const docSnap = await getDoc(docRef);
    
    if (!docSnap.exists()) {
      throw new Error('Analysis not found');
    }
    
    // Security check
    if (docSnap.data().userID !== userId) {
      throw new Error('Unauthorized');
    }
    
    await deleteDoc(docRef);
    console.log('✅ Analysis deleted from Firestore');
  } catch (error) {
    console.error('❌ Error deleting analysis:', error);
    throw error;
  }
};

/**
 * Get analysis summary stats for dashboard
 * @param {string} userId - Firebase Auth user ID
 * @returns {Promise<object>} - Stats object
 */
export const getAnalysisStats = async (userId) => {
  try {
    const analyses = await getUserAnalyses(userId);
    
    let totalAnalyses = analyses.length;
    let highRiskCount = 0;
    let mediumRiskCount = 0;
    let lowRiskCount = 0;
    let totalClauses = 0;
    
    analyses.forEach(analysis => {
      if (analysis.analysisType === 'detailed' && analysis.clauses) {
        analysis.clauses.forEach(clause => {
          totalClauses++;
          if (clause.risk_level === 'High') highRiskCount++;
          else if (clause.risk_level === 'Medium') mediumRiskCount++;
          else if (clause.risk_level === 'Low') lowRiskCount++;
        });
      } else if (analysis.risks) {
        analysis.risks.forEach(risk => {
          totalClauses++;
          if (risk.severity === 'High') highRiskCount++;
          else if (risk.severity === 'Medium') mediumRiskCount++;
          else lowRiskCount++;
        });
      }
    });
    
    return {
      totalAnalyses,
      highRiskCount,
      mediumRiskCount,
      lowRiskCount,
      totalClauses
    };
  } catch (error) {
    console.error('❌ Error calculating stats:', error);
    return {
      totalAnalyses: 0,
      highRiskCount: 0,
      mediumRiskCount: 0,
      lowRiskCount: 0,
      totalClauses: 0
    };
  }
};
// ============================================
// ADD THESE FUNCTIONS TO firestoreService.js
// ============================================
// At the end of your firestoreService.js, REPLACE the translation functions with these:

/**
 * Get translated content for an analysis
 * @param {string} analysisId - Document ID
 * @param {string} language - Language code (hi, es, fr)
 * @returns {Promise<object|null>} - Translated content or null
 */
export const getTranslation = async (analysisId, language) => {
  try {
    if (!db) {
      console.warn('⚠️ Firestore not configured');
      return null;
    }

    console.log(`🔍 Checking Firestore for ${language} translation of ${analysisId}`);
    
    const docRef = doc(db, COLLECTION_NAME, analysisId);
    const docSnap = await getDoc(docRef);
    
    if (!docSnap.exists()) {
      console.error('❌ Analysis not found in Firestore');
      throw new Error('Analysis not found');
    }
    
    const data = docSnap.data();
    const translations = data.translations || {};
    
    if (translations[language]) {
      console.log(`✅ Cached translation found for ${language}`);
      return translations[language];
    }
    
    console.log(`📝 No cached translation found for ${language}`);
    return null;
  } catch (error) {
    console.error('❌ Error fetching translation from Firestore:', error);
    return null; // Return null to allow fallback to API
  }
};

/**
 * Request translation from backend and save to Firestore
 * @param {string} analysisId - Document ID
 * @param {string} language - Language code
 * @param {string} userId - Firebase Auth user ID
 * @returns {Promise<object>} - Translated content
 */
export const requestTranslation = async (analysisId, language, userId) => {
  try {
    console.log(`🔄 Requesting translation to ${language}`, {
      analysisId,
      language,
      userId
    });
    
    // Build URL with query parameters (matching your backend)
    const url = `http://localhost:8000/translate/${analysisId}?language=${language}&user_id=${userId}`;
    console.log('📡 API URL:', url);
    
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
    });
    
    console.log('📡 Response status:', response.status);
    console.log('📡 Response ok:', response.ok);
    
    if (!response.ok) {
      let errorMessage;
      try {
        const errorData = await response.json();
        errorMessage = errorData.detail || errorData.message || 'Unknown error';
        console.error('❌ Backend error data:', errorData);
      } catch {
        errorMessage = await response.text();
        console.error('❌ Backend error text:', errorMessage);
      }
      throw new Error(`Backend error: ${response.status} - ${errorMessage}`);
    }
    
    const data = await response.json();
    console.log('📦 Full backend response:', JSON.stringify(data, null, 2));
    
    // Validate response structure
    if (!data) {
      throw new Error('Backend returned empty response');
    }
    
    // Check if backend returned an error
    if (data.error) {
      throw new Error(data.error);
    }
    
    // Extract translation - your backend returns { translated_content: {...} }
    const translation = data.translated_content || data.translation || data;
    console.log('📦 Extracted translation:', JSON.stringify(translation, null, 2));
    
    if (!translation || typeof translation !== 'object') {
      console.error('❌ Invalid translation type:', typeof translation);
      console.error('❌ Translation value:', translation);
      throw new Error('Invalid translation structure received from backend');
    }
    
    // Validate that translation has content
    const hasContent = translation.summary || translation.risks || translation.clauses;
    console.log('📊 Translation content check:', {
      hasSummary: !!translation.summary,
      hasRisks: !!translation.risks,
      hasClauses: !!translation.clauses,
      hasAnyContent: hasContent
    });
    
    if (!hasContent) {
      console.error('❌ Translation structure:', Object.keys(translation));
      throw new Error('Translation data is empty or incomplete');
    }
    
    console.log('✅ Translation validated successfully');
    return translation;
    
  } catch (error) {
    console.error('❌ requestTranslation failed:', error);
    console.error('❌ Error stack:', error.stack);
    throw error;
  }
};

/**
 * Get supported languages for translation
 * @returns {Promise<Array>} - Array of language objects
 */
export const getSupportedLanguages = async () => {
  try {
    const response = await fetch('http://localhost:8000/supported-languages');
    
    if (!response.ok) {
      throw new Error('Failed to fetch supported languages');
    }
    
    const data = await response.json();
    return data.languages;
  } catch (error) {
    console.error('❌ Error fetching supported languages:', error);
    // Return default languages if API fails
    return [
      { code: 'en', name: 'English' },
      { code: 'hi', name: 'Hindi' },
      { code: 'es', name: 'Spanish' },
      { code: 'fr', name: 'French' },
      { code: 'de', name: 'German' },
      { code: 'zh', name: 'Chinese' },
    ];
  }
};

const JOBS_COLLECTION = 'analysisJobs';

/**
 * Subscribe to real-time job status updates
 * 
 * @param {string} jobId - The job ID to monitor
 * @param {function} callback - Callback function called with job data on updates
 * @returns {function} Unsubscribe function
 * 
 * Usage:
 * const unsubscribe = subscribeToJobStatus(jobId, (jobData) => {
 *   console.log('Job status:', jobData.status);
 *   if (jobData.status === 'completed') {
 *     // Fetch results
 *   }
 * });
 * 
 * // Later: unsubscribe();
 */
export const subscribeToJobStatus = (jobId, callback) => {
  try {
    const jobRef = doc(db, JOBS_COLLECTION, jobId);
    
    const unsubscribe = onSnapshot(
      jobRef,
      (docSnapshot) => {
        if (docSnapshot.exists()) {
          const jobData = {
            id: docSnapshot.id,
            ...docSnapshot.data(),
            // Convert Firestore timestamps to JS dates
            createdAt: docSnapshot.data().createdAt?.toDate(),
            updatedAt: docSnapshot.data().updatedAt?.toDate(),
            startedAt: docSnapshot.data().startedAt?.toDate(),
            completedAt: docSnapshot.data().completedAt?.toDate(),
          };
          callback(jobData);
        } else {
          console.error('Job not found:', jobId);
          callback(null);
        }
      },
      (error) => {
        console.error('Error listening to job status:', error);
        callback(null);
      }
    );
    
    return unsubscribe;
  } catch (error) {
    console.error('❌ Error setting up job listener:', error);
    return () => {}; // Return no-op function
  }
};

/**
 * Get a single job by ID (one-time fetch)
 * 
 * @param {string} jobId - The job ID
 * @returns {Promise<object|null>} Job data or null if not found
 */
export const getJobById = async (jobId) => {
  try {
    const jobRef = doc(db, JOBS_COLLECTION, jobId);
    const jobSnapshot = await getDoc(jobRef);
    
    if (!jobSnapshot.exists()) {
      console.error('Job not found:', jobId);
      return null;
    }
    
    const data = jobSnapshot.data();
    return {
      id: jobSnapshot.id,
      ...data,
      createdAt: data.createdAt?.toDate(),
      updatedAt: data.updatedAt?.toDate(),
      startedAt: data.startedAt?.toDate(),
      completedAt: data.completedAt?.toDate(),
    };
  } catch (error) {
    console.error('❌ Error fetching job:', error);
    return null;
  }
};

/**
 * Get all jobs for a user
 * 
 * @param {string} userId - Firebase Auth user ID
 * @returns {Promise<Array>} Array of job documents
 */
export const getUserJobs = async (userId) => {
  try {
    const q = query(
      collection(db, JOBS_COLLECTION),
      where('userID', '==', userId),
      orderBy('createdAt', 'desc')
    );
    
    const querySnapshot = await getDocs(q);
    const jobs = [];
    
    querySnapshot.forEach((doc) => {
      jobs.push({
        id: doc.id,
        ...doc.data(),
        createdAt: doc.data().createdAt?.toDate(),
        updatedAt: doc.data().updatedAt?.toDate(),
        startedAt: doc.data().startedAt?.toDate(),
        completedAt: doc.data().completedAt?.toDate(),
      });
    });
    
    console.log(`✅ Retrieved ${jobs.length} jobs for user`);
    return jobs;
  } catch (error) {
    console.error('❌ Error fetching user jobs:', error);
    return [];
  }
};

/**
 * Get jobs by status for a user
 * 
 * @param {string} userId - Firebase Auth user ID
 * @param {string} status - Job status (pending, processing, completed, failed)
 * @returns {Promise<Array>} Array of job documents
 */
export const getUserJobsByStatus = async (userId, status) => {
  try {
    const q = query(
      collection(db, JOBS_COLLECTION),
      where('userID', '==', userId),
      where('status', '==', status),
      orderBy('createdAt', 'desc')
    );
    
    const querySnapshot = await getDocs(q);
    const jobs = [];
    
    querySnapshot.forEach((doc) => {
      jobs.push({
        id: doc.id,
        ...doc.data(),
        createdAt: doc.data().createdAt?.toDate(),
        updatedAt: doc.data().updatedAt?.toDate(),
      });
    });
    
    return jobs;
  } catch (error) {
    console.error('❌ Error fetching jobs by status:', error);
    return [];
  }
};

/**
 * Delete a job
 * 
 * @param {string} jobId - Job document ID
 * @param {string} userId - Firebase Auth user ID (for security)
 */
export const deleteJob = async (jobId, userId) => {
  try {
    const jobRef = doc(db, JOBS_COLLECTION, jobId);
    const jobSnap = await getDoc(jobRef);
    
    if (!jobSnap.exists()) {
      throw new Error('Job not found');
    }
    
    // Security check
    if (jobSnap.data().userID !== userId) {
      throw new Error('Unauthorized');
    }
    
    await deleteDoc(jobRef);
    console.log('✅ Job deleted from Firestore');
  } catch (error) {
    console.error('❌ Error deleting job:', error);
    throw error;
  }
};

/**
 * Get job statistics for dashboard
 * 
 * @param {string} userId - Firebase Auth user ID
 * @returns {Promise<object>} Job stats
 */
export const getJobStats = async (userId) => {
  try {
    const jobs = await getUserJobs(userId);
    
    const stats = {
      total: jobs.length,
      pending: jobs.filter(j => j.status === 'pending').length,
      processing: jobs.filter(j => j.status === 'processing').length,
      completed: jobs.filter(j => j.status === 'completed').length,
      failed: jobs.filter(j => j.status === 'failed').length,
    };
    
    return stats;
  } catch (error) {
    console.error('❌ Error calculating job stats:', error);
    return { total: 0, pending: 0, processing: 0, completed: 0, failed: 0 };
  }
};