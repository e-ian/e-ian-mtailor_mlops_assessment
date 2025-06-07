import requests
import os
import sys
import argparse

def test_prediction(api_url, api_key, image_path):
    """Test single image prediction and return class ID."""
    try:
        session = requests.Session()
        if api_key:
            session.headers.update({
                'Authorization': f'Bearer {api_key}'
            })
        
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
            headers = dict(session.headers)
            if 'Content-Type' in headers:
                del headers['Content-Type']
            
            response = requests.post(f"{api_url}/predict", files=files, headers=headers, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            # Handle both possible field names
            class_id = data.get('class_id') or data.get('predicted_class')
            confidence = data.get('confidence', 0)
            
            print(f"Prediction successful")
            print(f"Class ID: {class_id}")
            print(f"Confidence: {confidence:.4f}")
            return True
        else:
            print(f"Prediction failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"Prediction error: {e}")
        return False

def test_health(api_url, api_key):
    """Test health endpoint."""
    # Try different URL formats
    url_variants = [
        api_url,
        api_url.rstrip('/'),
        api_url + 'health',
        api_url.rstrip('/') + '/health',
    ]
    
    for url in url_variants:
        try:
            print(f"Trying health check: {url}")
            session = requests.Session()
            if api_key:
                session.headers.update({
                    'Authorization': f'Bearer {api_key}'
                }) 
            response = session.get(f"{url}/health" if not url.endswith('/health') else url, timeout=10)
            
            print(f"  Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                model_loaded = data.get('model_loaded', False)
                print(f"Health check: {'Model loaded' if model_loaded else 'Model not loaded'}")
                print(f"Working URL: {url}")
                return model_loaded, url
            else:
                print(f"  Response: {response.text[:100]}")
                
        except Exception as e:
            print(f"  Error: {e}")
    
    print(f"All health check attempts failed")
    return False, None

def main():
    parser = argparse.ArgumentParser(description='Test Cerebrium deployment')
    parser.add_argument('--api_url', required=True, help='API URL')
    parser.add_argument('--api_key', help='API key')
    parser.add_argument('--image_path', help='Test image path')
    
    args = parser.parse_args()
    
    print(f"Testing: {args.api_url}")
    print("=" * 50)
    health_ok, working_url = test_health(args.api_url, args.api_key)
    if health_ok and working_url and args.image_path and os.path.exists(args.image_path):
        prediction_ok = test_prediction(working_url, args.api_key, args.image_path)
        success = health_ok and prediction_ok
    elif args.image_path and os.path.exists(args.image_path):
        print("Trying prediction with original URL...")
        prediction_ok = test_prediction(args.api_url, args.api_key, args.image_path)
        success = prediction_ok
    else:
        print("No test image provided or health check failed")
        success = health_ok
    print("=" * 50)
    print(f"Result: {'PASS' if success else 'FAIL'}")
    sys.exit(0 if success else 1)
if __name__ == "__main__":
    main()
