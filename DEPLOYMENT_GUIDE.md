# 🚀 Deployment Guide - AI Placement Predictor

## Quick Deploy to Streamlit Cloud (FREE) ⭐ Recommended

### Prerequisites
1. GitHub account
2. Your app files ready

### Step-by-Step Instructions

#### 1. Create GitHub Repository

```bash
# Initialize git in your project folder
cd C:\Users\Nikhil\Desktop\AIDS-2
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit - AI Placement Predictor"

# Create a new repository on GitHub (go to github.com)
# Then connect and push:
git remote add origin https://github.com/YOUR_USERNAME/placement-predictor.git
git branch -M main
git push -u origin main
```

#### 2. Deploy on Streamlit Cloud

1. Go to **https://share.streamlit.io/**
2. Click **"New app"**
3. Connect your GitHub account
4. Select:
   - Repository: `YOUR_USERNAME/placement-predictor`
   - Branch: `main`
   - Main file path: `professional_app.py`
5. Click **"Deploy!"**

⏱️ Deployment takes 2-3 minutes

✅ You'll get a URL like: `https://YOUR_APP_NAME.streamlit.app`

---

## Alternative: Deploy to Hugging Face Spaces (FREE)

### Step 1: Create Hugging Face Account
- Go to https://huggingface.co/join
- Sign up for free

### Step 2: Create New Space
1. Go to https://huggingface.co/spaces
2. Click **"Create new Space"**
3. Name: `placement-predictor`
4. License: `mit`
5. SDK: **Streamlit**
6. Click **"Create Space"**

### Step 3: Upload Files
Upload these files to your Space:
- `professional_app.py`
- `requirements.txt`
- `rf_model_engineered.joblib`
- `ada_model_engineered.joblib`
- `hybrid_model.h5`
- `tabular_scaler.joblib`
- `sample_resume.txt`
- `.streamlit/config.toml`

✅ Your app will be live at: `https://huggingface.co/spaces/YOUR_USERNAME/placement-predictor`

---

## Alternative: Deploy to Render (FREE)

### Step 1: Create Render Account
- Go to https://render.com
- Sign up with GitHub

### Step 2: Create Web Service
1. Click **"New +"** → **"Web Service"**
2. Connect your GitHub repository
3. Settings:
   - Name: `placement-predictor`
   - Environment: `Python 3`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run professional_app.py --server.port $PORT --server.address 0.0.0.0`
4. Click **"Create Web Service"**

✅ Your app will be live at: `https://placement-predictor.onrender.com`

---

## Local Network Hosting (For LAN Access)

If you want to share with people on your local network:

```bash
# Run with network access
streamlit run professional_app.py --server.address 0.0.0.0

# Share this URL with others on your network:
# http://YOUR_LOCAL_IP:8501
# Example: http://192.168.0.102:8501
```

To find your local IP:
```bash
# Windows
ipconfig

# Look for "IPv4 Address" under your active connection
```

---

## Custom Domain Setup (After Deployment)

### For Streamlit Cloud:
1. Go to your app settings
2. Click **"Custom subdomain"**
3. Enter your desired name: `placement-ai`
4. You'll get: `https://placement-ai.streamlit.app`

### For Custom Domain (yourdomain.com):
1. Buy domain from Namecheap/GoDaddy
2. Add CNAME record pointing to your Streamlit/Hugging Face/Render URL
3. Configure in platform settings

---

## Files Needed for Deployment

✅ **Essential Files:**
```
professional_app.py          # Main app
requirements.txt             # Dependencies
rf_model_engineered.joblib   # Random Forest model
ada_model_engineered.joblib  # AdaBoost model
hybrid_model.h5              # Hybrid Neural Network
tabular_scaler.joblib        # Scaler for hybrid model
sample_resume.txt            # Sample resume
```

✅ **Optional but Recommended:**
```
README.md                    # Project documentation
.streamlit/config.toml       # Theme configuration
.gitignore                   # Git ignore file
```

---

## Environment Variables (If Needed)

If you need to add secrets:

### Streamlit Cloud:
1. Go to app settings
2. Click **"Secrets"**
3. Add in TOML format:
```toml
[secrets]
api_key = "your-key"
```

### Render:
1. Go to Environment tab
2. Add key-value pairs

---

## Cost Comparison

| Platform | Free Tier | Custom Domain | Limits |
|----------|-----------|---------------|---------|
| **Streamlit Cloud** | ✅ Yes | ✅ Subdomain | 1 private app |
| **Hugging Face** | ✅ Yes | ✅ Subdomain | Unlimited public |
| **Render** | ✅ Yes | ✅ Full support | Sleeps after 15min idle |

---

## Recommended: Streamlit Cloud

**Why?**
- ✅ Built specifically for Streamlit apps
- ✅ Fastest deployment (1-click)
- ✅ Auto-updates from GitHub
- ✅ Free subdomain
- ✅ Always on (no sleep)
- ✅ Good performance

---

## Post-Deployment Checklist

After deployment, verify:

- ✅ All 3 models load correctly
- ✅ Resume upload works (PDF & TXT)
- ✅ Predictions generate successfully
- ✅ Charts and visualizations display
- ✅ History tracking works
- ✅ Download CSV works
- ✅ Mobile responsive

---

## Troubleshooting

### "Module not found" error
- Check `requirements.txt` has all dependencies
- Redeploy after updating requirements

### "Model file not found"
- Ensure all `.joblib` and `.h5` files are in repository
- Check file names match exactly

### "App crashes on startup"
- Check Streamlit Cloud logs
- Look for missing dependencies
- Verify TensorFlow compatibility

### Slow loading
- Large model files (hybrid_model.h5) take time
- First load is slower, then cached

---

## Performance Tips

1. **Optimize Model Files:**
   - Compress models if > 100MB
   - Consider using `model_selection` to load on-demand

2. **Cache Data:**
   ```python
   @st.cache_resource
   def load_model():
       return joblib.load('model.joblib')
   ```

3. **Async Loading:**
   - Already implemented with spinners

---

## Share Your App! 🎉

Once deployed, share your link:
- Add to LinkedIn profile
- Include in resume
- Share on GitHub README
- Post on Twitter/social media

**Example:**
```
Check out my AI Placement Predictor! 🎓
Upload your resume and get instant placement predictions with detailed insights.

🔗 https://placement-ai.streamlit.app
```

---

## Need Help?

- Streamlit Docs: https://docs.streamlit.io/
- Community Forum: https://discuss.streamlit.io/
- GitHub Issues: Create issue in your repo

---

**Ready to deploy? Follow the Streamlit Cloud steps above! 🚀**
