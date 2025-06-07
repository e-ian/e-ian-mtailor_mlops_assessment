## MLOps
This project deploys a PyTorch ImageNet classification model on Cerebrium's serverless GPU platform using Docker and ONNX for optimized inference.

### Prerequisites
- Python 3.12+
- Docker
- Git
- Setup a cerebrium account and generate API_KEY

### Complete Setup

```bash
# 1. Clone and setup
git clone https://github.com/e-ian/e-ian-mtailor_mlops_assessment
cd mtailor_mlops_assessment
python -m venv tenv
source venv/bin/activate  # For Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Download model weights and samples
python scripts/download_weights.py

# 3. Setup environment
cp .env.example .env
# Edit .env and add CEREBRIUM_API_KEY provided

# 4. Convert to ONNX
python src/convert_to_onnx.py

# 5. Test locally
python tests/test.py

# 6. Deploy to Cerebrium
pip install cerebium

cerebrium login 
cerebrium deploy --disable-syntax-check --disable-confirmation # to skip interactive mode

# 7. Test deployment (replace with actual URL)
python tests/test_server.py --api_url "https://your-url" --api_key "<CEREBRIUM_API_KEY>" --image_path sample_images/n01440764_tench.jpeg
```

### Automated Deployment with GitHub Actions

This project includes CI/CD pipeline that automatically deploys to Cerebrium:

- **Auto-deployment**: Pushes to `main` or `develop` branches trigger automatic build everytime a commit is pushed to repo
- **Testing**: Runs tests and Docker builds before deployment
- **Setup**: Add `CEREBRIUM_API_KEY` as a GitHub repository secret

To enable auto-builds on github actions:
1. Go to your GitHub repo → Settings → Secrets and variables → Actions
2. Add `CEREBRIUM_API_KEY` with your Cerebrium API key
3. Push to `main` or `develop` branch - deployment happens automatically!

### Manual Deployment

For manual deployment, follow steps 7-8 above.

### Getting your Cerebrium API Key:
1. Sign up at [cerebrium.ai](https://cerebrium.ai)
2. Go to Dashboard → Settings → API Keys
3. Copy your API key to `.env` file (and GitHub secrets for auto-deployment)

### After Deployment:
1. Get your deployment URL from Cerebrium dashboard
2. Add it to `.env` as `API_URL=https://your-deployment-url`
3. Test with: `python test_server.py --api_url API_URL --api_key API_KEY --image_path sample_images/IMAGE.jpeg`
