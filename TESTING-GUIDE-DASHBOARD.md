# 🧪 LexiGuard Dashboard - Testing & Verification Guide

## 📋 Pre-Testing Checklist

Before you begin testing, ensure:

- [ ] Firebase project is created
- [ ] Firestore is enabled
- [ ] Security rules are deployed
- [ ] Frontend `.env` file is configured with Firebase credentials
- [ ] Dependencies are installed: `npm install firebase`
- [ ] Backend is running: `python -m uvicorn main:app --reload --port 8000`
- [ ] Frontend is running: `npm start`

---

## 🧪 Test Suite 1: Basic Flow (15 minutes)

### Test 1.1: User Authentication
**Objective**: Verify Firebase Authentication works

**Steps**:
1. Navigate to `http://localhost:3000`
2. Click "Login" button in navigation
3. Select Google account
4. Verify redirect back to home page
5. Check navigation bar shows user profile photo/name

**Expected Result**: ✅ User is authenticated and profile appears in nav

---

### Test 1.2: Document Upload & Analysis
**Objective**: Verify analysis works and saves to Firestore

**Steps**:
1. Navigate to `/upload`
2. Select "Detailed Analysis" (recommended)
3. Upload `sample_contract.txt` or any PDF/DOCX file
4. Click "Analyze Document"
5. Wait for analysis (15-30 seconds)
6. Verify redirect to `/results/:analysisId` (check URL has ID)

**Expected Result**: ✅ Analysis completes and URL contains analysisId

---

### Test 1.3: Results Page Display
**Objective**: Verify all analysis components render

**Steps**:
1. On Results page, verify these sections appear:
   - [ ] Document title and filename
   - [ ] Privacy notice badge (green banner)
   - [ ] Summary section (if available)
   - [ ] Risk/Clause cards
   - [ ] Chat sidebar
   - [ ] Stats sidebar

**Expected Result**: ✅ All sections render correctly

---

### Test 1.4: Dashboard Access
**Objective**: Verify dashboard displays saved analysis

**Steps**:
1. Click "Dashboard" link in navigation
2. Verify page loads successfully
3. Check stats cards show:
   - Total Analyses: 1 (or more)
   - Risk counts (High/Medium/Low)
4. Verify analysis card appears with:
   - Document title
   - Original filename
   - Upload date/time
   - Risk badges
   - "PII Redacted" badge (green)

**Expected Result**: ✅ Dashboard shows your analysis

---

## 🧪 Test Suite 2: Dashboard Features (20 minutes)

### Test 2.1: Search Functionality
**Objective**: Verify search filters analyses

**Steps**:
1. On Dashboard, upload 2-3 different documents (to have multiple analyses)
2. In search box, type part of a document title
3. Verify only matching analyses appear
4. Clear search box
5. Verify all analyses reappear

**Expected Result**: ✅ Search filters analyses in real-time

---

### Test 2.2: Filter by Type
**Objective**: Verify type filter works

**Steps**:
1. Upload at least one "Standard" and one "Detailed" analysis
2. Select "Standard Analysis" from filter dropdown
3. Verify only standard analyses appear
4. Select "Detailed Analysis"
5. Verify only detailed analyses appear
6. Select "All Types"
7. Verify all analyses appear

**Expected Result**: ✅ Filter correctly shows selected type

---

### Test 2.3: Sort Functionality
**Objective**: Verify sorting changes order

**Steps**:
1. Ensure you have 3+ analyses with different:
   - Upload times
   - Titles (alphabetically different)
   - Risk levels (mix of high/medium/low)
2. Select "Sort by Date" → Verify newest first
3. Select "Sort by Title" → Verify alphabetical order
4. Select "Sort by Risk Level" → Verify high risk first

**Expected Result**: ✅ Order changes based on sort selection

---

### Test 2.4: Rename Analysis
**Objective**: Verify inline title editing works

**Steps**:
1. Hover over any analysis card title
2. Click the edit icon (pencil) that appears
3. Type a new title: "Test Renamed Contract"
4. Click the green checkmark to save
5. Verify title updates immediately
6. Refresh page
7. Verify new title persists

**Expected Result**: ✅ Title is updated and saved to Firestore

---

### Test 2.5: Star/Favorite Analysis
**Objective**: Verify starring works and persists

**Steps**:
1. Click the star icon on any analysis card
2. Verify star fills with yellow color
3. Refresh page
4. Verify star remains filled
5. Select "Starred" from filter dropdown
6. Verify only starred analyses appear
7. Click star again to unstar
8. Verify star becomes outline only

**Expected Result**: ✅ Star toggle works and persists

---

### Test 2.6: Delete Analysis
**Objective**: Verify delete with confirmation

**Steps**:
1. Note the total count of analyses
2. Click trash icon on an analysis card
3. Verify confirmation modal appears with:
   - Warning message
   - "Delete" button (red)
   - "Cancel" button
4. Click "Cancel"
5. Verify analysis is still visible
6. Click trash icon again
7. Click "Delete"
8. Verify analysis disappears from dashboard
9. Verify stats cards update (total count decreases)

**Expected Result**: ✅ Analysis is deleted with confirmation

---

## 🧪 Test Suite 3: Navigation & Persistence (15 minutes)

### Test 3.1: View Analysis from Dashboard
**Objective**: Verify "View Analysis" loads from Firestore

**Steps**:
1. On Dashboard, click "View Analysis" on any card
2. Verify redirect to `/results/:analysisId` (check URL)
3. Verify analysis data loads correctly:
   - [ ] Summary appears
   - [ ] Risks/clauses appear
   - [ ] Chat is functional
4. Verify URL contains the analysisId

**Expected Result**: ✅ Full analysis loads from Firestore

---

### Test 3.2: Direct URL Access
**Objective**: Verify analysis loads from URL

**Steps**:
1. Copy the URL from a Results page (e.g., `/results/abc123def`)
2. Open a new browser tab
3. Paste the URL and press Enter
4. Verify analysis loads correctly
5. Try editing the analysisId to a random value
6. Verify error message or redirect to dashboard

**Expected Result**: ✅ Valid IDs load, invalid IDs show error

---

### Test 3.3: Browser Refresh
**Objective**: Verify data persists across refresh

**Steps**:
1. On Dashboard, note the analyses and stats
2. Press F5 or Ctrl+R to refresh
3. Verify all analyses still appear
4. Verify stats are unchanged
5. On Results page, refresh
6. Verify analysis data reloads correctly

**Expected Result**: ✅ All data persists after refresh

---

### Test 3.4: Multi-Tab Consistency
**Objective**: Verify changes sync (if using real-time listeners)

**Steps**:
1. Open Dashboard in two browser tabs (Tab A and Tab B)
2. In Tab A, rename an analysis
3. In Tab B, refresh page
4. Verify new title appears in Tab B
5. In Tab A, delete an analysis
6. In Tab B, refresh page
7. Verify analysis is gone in Tab B

**Expected Result**: ✅ Changes persist across tabs (after refresh)

---

## 🧪 Test Suite 4: Security & Privacy (10 minutes)

### Test 4.1: Firestore Data Inspection
**Objective**: Verify only redacted text is stored

**Steps**:
1. Upload a document with personal info (name, email, address)
2. After analysis, go to Firebase Console → Firestore
3. Find `userAnalyses` collection
4. Click on your analysis document
5. Check the `redactedDocumentText` field
6. Verify you see placeholders:
   - `[PERSON_NAME]`
   - `[EMAIL_ADDRESS]`
   - `[STREET_ADDRESS]`
   - `[PHONE_NUMBER]`
7. Verify you do NOT see original personal info

**Expected Result**: ✅ Only redacted text is stored

---

### Test 4.2: Privacy Badge Display
**Objective**: Verify PII redaction badge shows

**Steps**:
1. On Dashboard, check each analysis card
2. Verify green badge says "✓ PII Redacted"
3. On Results page, check privacy notice banner
4. Verify green banner says "✓ Your Personal Data Has Been Redacted for Privacy."

**Expected Result**: ✅ Privacy badges display prominently

---

### Test 4.3: User Isolation
**Objective**: Verify users can't access each other's data

**Steps**:
1. Login as User A
2. Upload and analyze a document
3. Note the analysisId from the URL
4. Logout
5. Login as User B (different Google account)
6. Try to navigate to User A's analysis URL: `/results/:analysisId`
7. Verify you get an error or redirect

**Expected Result**: ✅ User B cannot access User A's analysis

---

### Test 4.4: Firestore Security Rules Test
**Objective**: Verify security rules block unauthorized access

**Steps**:
1. Open browser DevTools (F12) → Console tab
2. Run this code:
```javascript
// Try to read another user's analysis
const db = firebase.firestore();
db.collection('userAnalyses')
  .where('userID', '!=', firebase.auth().currentUser.uid)
  .get()
  .then(snapshot => {
    if (snapshot.empty) {
      console.log('✅ Security rules working - no data returned');
    } else {
      console.error('❌ Security issue - got other user data!');
    }
  })
  .catch(error => {
    console.log('✅ Security rules blocked query:', error.message);
  });
```
3. Check console output

**Expected Result**: ✅ Query is blocked or returns empty

---

## 🧪 Test Suite 5: Edge Cases (15 minutes)

### Test 5.1: No Analyses (Empty State)
**Objective**: Verify empty state displays correctly

**Steps**:
1. Delete all your analyses (or create a new test account)
2. Navigate to Dashboard
3. Verify empty state shows:
   - [ ] Folder icon
   - [ ] "No analyses yet" message
   - [ ] "Analyze Your First Document" button
4. Click the button
5. Verify redirect to Upload page

**Expected Result**: ✅ Empty state is helpful and actionable

---

### Test 5.2: Long Document Titles
**Objective**: Verify UI handles long titles gracefully

**Steps**:
1. Rename an analysis to a very long title (100+ characters)
2. Verify title truncates on dashboard card (doesn't break layout)
3. Verify full title shows when editing
4. View the analysis
5. Verify title displays correctly on Results page

**Expected Result**: ✅ Long titles don't break UI

---

### Test 5.3: Special Characters in Title
**Objective**: Verify special characters are handled

**Steps**:
1. Rename an analysis with special characters: `Contract #1 (Urgent!) - Review @ 3PM`
2. Verify title saves correctly
3. Verify search works with special characters
4. Verify title displays correctly everywhere

**Expected Result**: ✅ Special characters are preserved

---

### Test 5.4: Network Interruption
**Objective**: Verify error handling for network issues

**Steps**:
1. Open DevTools → Network tab
2. Set throttling to "Offline"
3. Try to load Dashboard
4. Verify error message or loading state appears
5. Set back to "Online"
6. Refresh page
7. Verify Dashboard loads correctly

**Expected Result**: ✅ Graceful error handling

---

### Test 5.5: Large Number of Analyses
**Objective**: Verify performance with many analyses

**Steps**:
1. Upload 20+ documents (if feasible, or use test data)
2. Navigate to Dashboard
3. Verify page loads in < 3 seconds
4. Try search, filter, and sort
5. Verify no lag or freezing
6. Check browser console for Firestore warnings about missing indexes
7. Create indexes if prompted

**Expected Result**: ✅ Performs well with many analyses

---

## 🧪 Test Suite 6: Integration with Existing Features (10 minutes)

### Test 6.1: Chat with Document
**Objective**: Verify chat works with redacted text

**Steps**:
1. View an analysis from Dashboard
2. In chat sidebar, type: "What is the main purpose of this agreement?"
3. Wait for AI response
4. Verify response references the document (even with placeholders)
5. Ask: "Who is the landlord?"
6. Verify AI explains that specific details were redacted

**Expected Result**: ✅ Chat works with redacted document text

---

### Test 6.2: Negotiation Email Generation
**Objective**: Verify negotiation feature works

**Steps**:
1. View an analysis with risky clauses
2. Click "Draft Negotiation Email" on a clause
3. Verify email generates successfully
4. Check if email mentions redacted placeholders appropriately
5. Click "Copy to Clipboard"
6. Verify success message appears
7. Paste in a text editor
8. Verify email text copied correctly

**Expected Result**: ✅ Email generation works with redacted clauses

---

### Test 6.3: Document Email Generation
**Objective**: Verify comprehensive email feature works

**Steps**:
1. View an analysis from Dashboard
2. Click "Generate Document Email"
3. Verify email preview generates
4. Verify email summarizes all findings
5. Click "Copy Text"
6. Verify copy works
7. Close modal
8. Verify can re-open and regenerate

**Expected Result**: ✅ Document email works from saved analyses

---

## 🎯 Acceptance Criteria

### ✅ PASS Criteria
Your implementation passes if:

1. **Authentication**:
   - [ ] User can login with Google
   - [ ] Profile appears in navigation
   - [ ] Protected routes redirect unauthenticated users

2. **Upload & Save**:
   - [ ] Documents can be uploaded and analyzed
   - [ ] Analyses auto-save to Firestore
   - [ ] User is redirected to Results with analysisId

3. **Dashboard**:
   - [ ] All analyses display correctly
   - [ ] Stats cards show accurate counts
   - [ ] Privacy notice is prominent
   - [ ] Search, filter, and sort work
   - [ ] Can rename, star, and delete analyses
   - [ ] Empty state displays when no analyses

4. **Results Loading**:
   - [ ] Can load analysis from Dashboard (via URL)
   - [ ] All existing features work (chat, negotiation, email)
   - [ ] Privacy badge displays

5. **Security & Privacy**:
   - [ ] Only redacted text is stored in Firestore
   - [ ] Users cannot access each other's analyses
   - [ ] Firestore rules block unauthorized access

6. **Persistence**:
   - [ ] Data persists across browser refresh
   - [ ] Changes save to Firestore immediately
   - [ ] Can access saved analyses directly via URL

7. **Performance**:
   - [ ] Dashboard loads in < 3 seconds (< 50 analyses)
   - [ ] No console errors in browser
   - [ ] Firestore indexes are created

8. **Integration**:
   - [ ] Existing Upload functionality unchanged
   - [ ] Existing Results features all work
   - [ ] Authentication flow unchanged
   - [ ] Color theme consistent

---

## 🐛 Common Issues & Fixes

### Issue: Firebase not initialized
**Symptoms**: Console error "Firebase app not initialized"
**Fix**: 
1. Check `.env` file has all Firebase variables
2. Restart development server
3. Clear browser cache

### Issue: Permission denied in Firestore
**Symptoms**: Console error "Missing or insufficient permissions"
**Fix**:
1. Verify Firestore Security Rules are deployed
2. Check user is authenticated (profile photo in nav)
3. Verify `userID` in document matches your Firebase UID

### Issue: Analyses not appearing
**Symptoms**: Dashboard shows empty state but Firestore has data
**Fix**:
1. Check browser console for errors
2. Verify Firestore query isn't failing
3. Create missing indexes if prompted
4. Check `userID` field in Firestore matches your UID

### Issue: Can't load analysis from URL
**Symptoms**: Results page shows "No Analysis Data"
**Fix**:
1. Verify URL format: `/results/:analysisId` (with ID)
2. Check analysisId exists in Firestore
3. Verify you own the analysis (security rule)
4. Check network tab for 403/404 errors

---

## 📊 Test Results Template

Use this to track your testing:

```
LexiGuard Dashboard Test Results
Date: _______________
Tester: _______________

Test Suite 1: Basic Flow
  [✅/❌] 1.1 User Authentication
  [✅/❌] 1.2 Document Upload & Analysis
  [✅/❌] 1.3 Results Page Display
  [✅/❌] 1.4 Dashboard Access

Test Suite 2: Dashboard Features
  [✅/❌] 2.1 Search Functionality
  [✅/❌] 2.2 Filter by Type
  [✅/❌] 2.3 Sort Functionality
  [✅/❌] 2.4 Rename Analysis
  [✅/❌] 2.5 Star/Favorite
  [✅/❌] 2.6 Delete Analysis

Test Suite 3: Navigation & Persistence
  [✅/❌] 3.1 View Analysis from Dashboard
  [✅/❌] 3.2 Direct URL Access
  [✅/❌] 3.3 Browser Refresh
  [✅/❌] 3.4 Multi-Tab Consistency

Test Suite 4: Security & Privacy
  [✅/❌] 4.1 Firestore Data Inspection
  [✅/❌] 4.2 Privacy Badge Display
  [✅/❌] 4.3 User Isolation
  [✅/❌] 4.4 Security Rules Test

Test Suite 5: Edge Cases
  [✅/❌] 5.1 No Analyses (Empty State)
  [✅/❌] 5.2 Long Document Titles
  [✅/❌] 5.3 Special Characters
  [✅/❌] 5.4 Network Interruption
  [✅/❌] 5.5 Large Number of Analyses

Test Suite 6: Integration
  [✅/❌] 6.1 Chat with Document
  [✅/❌] 6.2 Negotiation Email
  [✅/❌] 6.3 Document Email

Overall Result: [PASS / FAIL / NEEDS WORK]
Notes: _______________________________________________
```

---

## 🎉 Testing Complete!

If all tests pass:
- ✅ Your LexiGuard Dashboard is production-ready!
- ✅ Privacy-first architecture is working
- ✅ User experience is excellent
- ✅ Security is properly implemented

Next steps:
1. Deploy to production (Firebase Hosting + Cloud Run)
2. Monitor usage (Firebase Analytics)
3. Gather user feedback
4. Plan next features (tags, notes, sharing)

**Happy Testing! 🚀**
