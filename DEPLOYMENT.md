# Render Deployment Instructions

## Quick Start Guide - Deploy in 15 Minutes

Your polymer-solubility-app is ready for deployment! Follow these steps to get your app live on Render.

---

## Prerequisites

- ✅ GitHub repository: `aaltamimi2/RAG-polymer-solubility`
- ✅ Production branch: `production` (already pushed)
- ✅ Google Gemini API key (you already have this)

---

## Step 1: Create Render Account (2 minutes)

1. Go to **https://render.com**
2. Click **"Get Started for Free"**
3. Choose **"Sign up with GitHub"** (recommended)
4. Authorize Render to access your GitHub repositories
5. Complete the signup process

---

## Step 2: Create New Web Service (3 minutes)

1. In Render dashboard, click **"New +"** (top right)
2. Select **"Web Service"**
3. Click **"Connect account"** if prompted to connect GitHub
4. Find your repository: `aaltamimi2/RAG-polymer-solubility`
5. Click **"Connect"**

---

## Step 3: Configure the Service (5 minutes)

Render will show a configuration page. Set these values:

### Basic Settings:
- **Name**: `polymer-solubility-app` (or your preferred name)
- **Region**: Select closest to you (e.g., Oregon, Ohio, Frankfurt)
- **Branch**: `production` ⚠️ IMPORTANT: Select the production branch!
- **Root Directory**: Leave blank
- **Runtime**: Should auto-detect as **Python**

### Build & Deploy Settings:
Render should auto-detect from `render.yaml`. Verify:
- **Build Command**: `pip install -r requirements.txt && cd frontend && npm install && npm run build && cd ..`
- **Start Command**: `python app_server.py`

### Plan:
- Select **"Free"** tier

---

## Step 4: Set Environment Variables (3 minutes) - CRITICAL FOR SECURITY

This is where your Google API key is protected!

1. Scroll down to **"Environment Variables"**
2. Click **"Add Environment Variable"**
3. Add your Google API key:
   - **Key**: `GOOGLE_API_KEY`
   - **Value**: `[Your actual Google Gemini API key]`
   - ⚠️ **IMPORTANT**: Toggle the **"Secret"** switch to ON (this hides it from logs)

4. Verify these are set automatically from `render.yaml`:
   - `PORT` = `8000`
   - `PYTHON_VERSION` = `3.11.0`

---

## Step 5: Deploy! (5-10 minutes)

1. Click **"Create Web Service"** at the bottom
2. Render will now:
   - Clone your repository
   - Install Python dependencies (~2 min)
   - Install Node dependencies (~1 min)
   - Build React frontend (~1 min)
   - Start your server (~30 sec)
   - Assign you a public URL

3. Watch the build logs in real-time
4. Wait for the status to show **"Live"** (green indicator)

---

## Step 6: Test Your Deployment (2 minutes)

Once deployment is complete:

1. Render will show your public URL: `https://polymer-solubility-app.onrender.com`
2. Click the URL to open your app
3. Test the functionality:
   - Click "List Polymers" button
   - Try a query: "Find solvents for HDPE"
   - Check the model selector (Flash Lite, Flash, Pro)

⚠️ **First Load**: The first request may take 30-60 seconds if the app was sleeping (free tier auto-sleeps after 15 min of inactivity)

---

## Step 7: Enable Auto-Deploy (2 minutes)

Set up automatic deployments when you push to GitHub:

1. In Render dashboard, go to your service
2. Click **"Settings"** tab
3. Scroll to **"Build & Deploy"** section
4. Find **"Auto-Deploy"**
5. Ensure it's set to **"Yes"** for the `production` branch

**Now**: Every time you push to the `production` branch, Render automatically redeploys!

---

## Your Deployment is Complete! 🎉

**Your app is now live at**: `https://polymer-solubility-app.onrender.com`

### What's Protected:
- ✅ Your Google API key is stored as a **secret** (not visible in logs or UI)
- ✅ Environment variables are encrypted at rest
- ✅ Never committed to Git (`.env` in `.gitignore`)
- ✅ HTTPS/SSL automatically enabled

### What's Next:
- Share the URL with colleagues
- Monitor usage in Render dashboard
- Check Gemini API usage in Google Cloud Console

---

## How to Deploy Updates

### Frontend Changes:
```bash
# Make your changes in frontend/src/
git add frontend/
git commit -m "Update UI"
git push origin production
# Auto-deploys in 3-5 minutes
```

### Backend Changes:
```bash
# Make changes to .py files
git add *.py
git commit -m "Add new feature"
git push origin production
# Auto-deploys in 3-5 minutes
```

### Both:
```bash
git add .
git commit -m "Full update"
git push origin production
# Auto-deploys in 5-10 minutes
```

---

## Monitoring & Costs

### Free Tier Limits:
- **Hosting**: FREE
- **Runtime**: 750 hours/month (always on, or auto-sleep after 15 min)
- **Memory**: 512 MB RAM
- **Bandwidth**: Unlimited

### Actual Costs:
- **Render**: $0/month (free tier)
- **Gemini API**: ~$3-15/month (depends on usage)
  - Flash Lite: $0.000075 per 1K input tokens
  - Flash: $0.00015 per 1K input tokens
  - Pro: More expensive (check Google pricing)

### Monitoring:
1. **Render Dashboard**: View logs, metrics, deployment history
2. **Gemini Console**: https://console.cloud.google.com - Monitor API usage
3. **Logs**: In Render dashboard → "Logs" tab

---

## Troubleshooting

### Issue: "Application failed to respond"
**Solution**:
- Check "Logs" tab in Render dashboard
- Look for CSV loading errors
- Verify all files were deployed correctly

### Issue: "Out of memory"
**Solution**:
- Upgrade to Render Starter plan ($7/month) for 512 MB → 2 GB RAM
- Or optimize memory usage

### Issue: "Slow first load (60+ seconds)"
**Explanation**: Free tier auto-sleeps after 15 min of inactivity
**Solutions**:
- Accept the trade-off (it's free!)
- Upgrade to paid plan ($7/month) for no sleep
- Use a service like UptimeRobot to ping every 14 minutes (keeps it awake)

### Issue: API key not working
**Solution**:
- Verify `GOOGLE_API_KEY` is set in Environment Variables
- Make sure it's marked as "Secret"
- Redeploy: Settings → Manual Deploy → "Clear build cache & deploy"

---

## Security Best Practices

✅ **Already Implemented**:
- API key stored as Render secret
- `.env` in `.gitignore`
- HTTPS/SSL enabled by default
- Dynamic API URL (no hardcoded endpoints)

🔜 **Recommended for Production**:
- Add rate limiting (10 requests/min per IP)
- Restrict CORS to your domain only
- Implement usage quotas per user
- Add monitoring/analytics

---

## Custom Domain (Optional)

Want `polymers.yourdomain.com` instead of `polymer-solubility-app.onrender.com`?

1. In Render dashboard → "Settings" → "Custom Domain"
2. Add your domain
3. Update DNS records as instructed
4. Render automatically provisions SSL certificate
5. **Cost**: FREE on free tier!

---

## Upgrade Options

If you need better performance:

### Render Starter ($7/month):
- No auto-sleep
- 512 MB RAM
- Faster CPU
- Priority support

### Google Cloud Run (Alternative):
- Pay-per-use (~$0-5/month)
- Faster cold starts (5-10s vs 60s)
- More scalable
- Requires Docker setup

---

## Support

- **Render Docs**: https://render.com/docs
- **Render Community**: https://community.render.com
- **This App's Docs**: See README.md for app-specific help

---

## Summary

✅ Production branch created and pushed to GitHub
✅ Render configuration complete (`render.yaml`)
✅ API key protection enabled (secrets)
✅ Auto-deploy configured
✅ Frontend/backend ready for deployment

**Next**: Follow Steps 1-7 above to deploy to Render!

**Estimated Time**: 15-20 minutes total
**Cost**: $0 for hosting, ~$3-15/month for Gemini API usage
