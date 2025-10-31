# Quick Deployment Script for Windows PowerShell

Write-Host "AI Placement Predictor - Deployment Setup" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""

# Check if git is installed
Write-Host "Checking prerequisites..." -ForegroundColor Yellow
try {
    git --version | Out-Null
    Write-Host "Git is installed" -ForegroundColor Green
} catch {
    Write-Host "Git is not installed. Please install Git first:" -ForegroundColor Red
    Write-Host "   Download from: https://git-scm.com/download/win" -ForegroundColor Yellow
    exit
}

Write-Host ""
Write-Host "Choose deployment option:" -ForegroundColor Cyan
Write-Host "1. Streamlit Cloud (Recommended - Free & Easy)" -ForegroundColor White
Write-Host "2. Hugging Face Spaces (Alternative - Free)" -ForegroundColor White
Write-Host "3. Setup GitHub repository only" -ForegroundColor White
Write-Host "4. Exit" -ForegroundColor White
Write-Host ""

$choice = Read-Host "Enter your choice (1-4)"

switch ($choice) {
    "1" {
        Write-Host ""
        Write-Host "Streamlit Cloud Deployment Steps:" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "1. First, we will setup your GitHub repository..." -ForegroundColor Yellow
        Write-Host ""
        
        # Get GitHub username
        $githubUser = Read-Host "Enter your GitHub username"
        $repoName = Read-Host "Enter repository name (e.g., placement-predictor)"
        
        Write-Host ""
        Write-Host "Initializing Git repository..." -ForegroundColor Yellow
        git init
        git add .
        git commit -m "Initial commit - AI Placement Predictor"
        
        Write-Host ""
        Write-Host "Git repository initialized" -ForegroundColor Green
        Write-Host ""
        Write-Host "Next Steps:" -ForegroundColor Cyan
        Write-Host "1. Go to GitHub.com and create a new repository named: $repoName" -ForegroundColor White
        Write-Host "2. Run these commands in PowerShell:" -ForegroundColor White
        Write-Host ""
        Write-Host "   git remote add origin https://github.com/$githubUser/$repoName.git" -ForegroundColor Yellow
        Write-Host "   git branch -M main" -ForegroundColor Yellow
        Write-Host "   git push -u origin main" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "3. Go to https://share.streamlit.io/" -ForegroundColor White
        Write-Host "4. Click New app" -ForegroundColor White
        Write-Host "5. Select your repository: $githubUser/$repoName" -ForegroundColor White
        Write-Host "6. Main file: professional_app.py" -ForegroundColor White
        Write-Host "7. Click Deploy!" -ForegroundColor White
        Write-Host ""
        Write-Host "Your app will be live in 2-3 minutes!" -ForegroundColor Green
    }
    
    "2" {
        Write-Host ""
        Write-Host "Hugging Face Spaces Deployment Steps:" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "1. Go to https://huggingface.co/join and create account" -ForegroundColor White
        Write-Host "2. Go to https://huggingface.co/spaces" -ForegroundColor White
        Write-Host "3. Click Create new Space" -ForegroundColor White
        Write-Host "4. Name: placement-predictor" -ForegroundColor White
        Write-Host "5. SDK: Streamlit" -ForegroundColor White
        Write-Host "6. Upload these files:" -ForegroundColor White
        Write-Host "   - professional_app.py" -ForegroundColor Yellow
        Write-Host "   - requirements.txt" -ForegroundColor Yellow
        Write-Host "   - All .joblib and .h5 model files" -ForegroundColor Yellow
        Write-Host "   - sample_resume.txt" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Your app will be live instantly!" -ForegroundColor Green
    }
    
    "3" {
        Write-Host ""
        Write-Host "Setting up GitHub repository..." -ForegroundColor Yellow
        
        $githubUser = Read-Host "Enter your GitHub username"
        $repoName = Read-Host "Enter repository name"
        
        git init
        git add .
        git commit -m "Initial commit - AI Placement Predictor"
        
        Write-Host ""
        Write-Host "Repository setup complete!" -ForegroundColor Green
        Write-Host ""
        Write-Host "Next steps:" -ForegroundColor Cyan
        Write-Host "1. Create repository on GitHub: $repoName" -ForegroundColor White
        Write-Host "2. Run:" -ForegroundColor White
        Write-Host ""
        Write-Host "   git remote add origin https://github.com/$githubUser/$repoName.git" -ForegroundColor Yellow
        Write-Host "   git branch -M main" -ForegroundColor Yellow
        Write-Host "   git push -u origin main" -ForegroundColor Yellow
    }
    
    "4" {
        Write-Host "Exiting..." -ForegroundColor Yellow
        exit
    }
    
    default {
        Write-Host "Invalid choice. Exiting..." -ForegroundColor Red
        exit
    }
}

Write-Host ""
Write-Host "For detailed instructions, see DEPLOYMENT_GUIDE.md" -ForegroundColor Cyan
Write-Host ""
pause
