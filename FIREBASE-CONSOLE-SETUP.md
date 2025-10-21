# 🔥 Firebase Console Setup Checklist

## IMPORTANT: Enable Google Authentication in Firebase Console

Before testing login, you MUST enable Google Sign-In in your Firebase Console:

### Step-by-Step Instructions:

1. **Go to Firebase Console**
   - Visit: https://console.firebase.google.com
   - Sign in with your Google account

2. **Select Your Project**
   - Click on project: **lexiguard-ce0ff**

3. **Navigate to Authentication**
   - In the left sidebar, click "**Build**" or "**Authentication**"
   - Click on the "**Sign-in method**" tab

4. **Enable Google Provider**
   - Find "**Google**" in the list of providers
   - Click on it to open settings
   - Toggle the "**Enable**" switch to ON
   - Add a **Project support email** (your email)
   - Click "**Save**"

5. **Add Authorized Domains** (if needed)
   - Still in "Sign-in method" tab
   - Scroll to "**Authorized domains**" section
   - Make sure these are listed:
     - `localhost` (should be there by default)
     - Add your production domain when deploying

6. **Verify Settings**
   - Google provider should show status: "**Enabled**"
   - You should see your support email

## ✅ Current Firebase Configuration

Your app is configured with:
```
Project ID: lexiguard-ce0ff
Auth Domain: lexiguard-ce0ff.firebaseapp.com
```

## 🔒 Optional: Firebase Security Rules

For production, set up Firestore/Storage rules:

### Firestore Rules (if using Firestore):
```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    // Only authenticated users can read/write
    match /{document=**} {
      allow read, write: if request.auth != null;
    }
  }
}
```

### Storage Rules (if using Storage):
```javascript
rules_version = '2';
service firebase.storage {
  match /b/{bucket}/o {
    match /{allPaths=**} {
      allow read, write: if request.auth != null;
    }
  }
}
```

## 🎯 Quick Verification

After enabling Google Sign-In:
1. Go to Firebase Console → Authentication → Users
2. This tab will show all logged-in users after they sign in
3. After your first login, you should see your account listed here

## ⚠️ Common Issues

### Issue: "This app is not authorized to use Firebase Authentication"
**Solution**: 
- Make sure Google provider is enabled
- Check that your domain is in authorized domains
- Verify API key matches in `.env` file

### Issue: "Popup closed by user"
**Solution**: 
- User closed the Google popup before completing sign-in
- This is normal behavior, just try again

### Issue: "Network request failed"
**Solution**: 
- Check internet connection
- Make sure Firebase project is active (not deleted)
- Verify firebaseConfig in `firebase/config.js`

## 📞 Need Help?

If you encounter issues:
1. Check browser console for error messages (F12)
2. Verify Google provider is enabled in Firebase Console
3. Check Network tab to see if requests are being made
4. Make sure `.env` file is in the correct location (lexiguard-frontend root)

---

**Next Steps**: 
1. ✅ Enable Google Sign-In in Firebase Console (see instructions above)
2. ✅ Start your development server: `npm start`
3. ✅ Test login at `http://localhost:3000/login`
4. ✅ Check if authentication works by accessing `/upload`
