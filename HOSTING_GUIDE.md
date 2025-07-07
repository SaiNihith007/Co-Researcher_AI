# 🌐 Hosting Guide for Multi-Agent Research System

This guide provides multiple options for hosting your Multi-Agent Research System online.

## 🚀 Quick Start - Local Testing

### 1. Test Locally First
```bash
# In your project directory
python app.py
```
Then open: http://localhost:5000

### 2. Test with Production Server
```bash
gunicorn app:app --bind 0.0.0.0:8000
```
Then open: http://localhost:8000

---

## ☁️ OPTION 1: Heroku (Easiest - Free Tier Available)

### Prerequisites
- Heroku account (free): https://signup.heroku.com
- Heroku CLI installed: https://devcenter.heroku.com/articles/heroku-cli

### Steps
```bash
# 1. Login to Heroku
heroku login

# 2. Create a new Heroku app
heroku create your-research-system-name

# 3. Set environment variables (optional)
heroku config:set SECRET_KEY=your-secret-key-here

# 4. Deploy
git add .
git commit -m "Deploy to Heroku"
git push heroku main

# 5. Open your app
heroku open
```

### Environment Variables for Heroku
```bash
# Optional: Set a custom secret key
heroku config:set SECRET_KEY=your-random-secret-key

# If you want to pre-configure OpenAI (users can still override)
heroku config:set DEFAULT_OPENAI_API_KEY=sk-your-key-here
```

---

## 🐳 OPTION 2: Docker (Most Portable)

### Create Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

### Build and Run
```bash
# Build the Docker image
docker build -t research-system .

# Run locally
docker run -p 5000:5000 research-system

# Or run with environment variables
docker run -p 5000:5000 -e SECRET_KEY=your-secret research-system
```

### Deploy to Cloud (Examples)

#### Google Cloud Run
```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/your-project/research-system

# Deploy to Cloud Run
gcloud run deploy --image gcr.io/your-project/research-system --platform managed
```

#### AWS ECS / Azure Container Instances
Similar process - build image and deploy to your preferred container service.

---

## 🖥️ OPTION 3: VPS/Server (Maximum Control)

### On Ubuntu/Debian Server
```bash
# 1. Update system
sudo apt update && sudo apt upgrade -y

# 2. Install Python and dependencies
sudo apt install python3 python3-pip python3-venv nginx supervisor -y

# 3. Create user for the app
sudo adduser research
sudo usermod -aG sudo research
su - research

# 4. Upload your code
# Use git clone or scp to get your code on the server

# 5. Setup virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 6. Configure Gunicorn service
sudo nano /etc/supervisor/conf.d/research-system.conf
```

#### Supervisor Config (`/etc/supervisor/conf.d/research-system.conf`)
```ini
[program:research-system]
command=/home/research/Project_ph1/.venv/bin/gunicorn app:app --bind 127.0.0.1:8000
directory=/home/research/Project_ph1
user=research
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/research-system.log
```

#### Nginx Config (`/etc/nginx/sites-available/research-system`)
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /static {
        alias /home/research/Project_ph1/static;
    }
}
```

#### Enable and Start Services
```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/research-system /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx

# Start supervisor
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start research-system
```

---

## 🔒 OPTION 4: Railway/Render (Modern Alternatives)

### Railway
1. Go to https://railway.app
2. Connect your GitHub repo
3. Deploy automatically

### Render
1. Go to https://render.com
2. Connect your GitHub repo
3. Choose "Web Service"
4. Build command: `pip install -r requirements.txt`
5. Start command: `gunicorn app:app`

---

## 🛡️ Security Considerations

### Environment Variables
Never commit sensitive data. Use environment variables:

```python
# In your app.py
import os
app.secret_key = os.environ.get('SECRET_KEY', 'default-dev-key')
```

### Production Settings
```bash
# Set these environment variables in production
export SECRET_KEY="your-random-secret-key"
export FLASK_ENV="production"
export OPENAI_API_KEY="sk-your-key"  # Optional default
```

### SSL/HTTPS
- **Heroku**: Automatic HTTPS
- **Railway/Render**: Automatic HTTPS  
- **VPS**: Use Let's Encrypt/Certbot
- **Docker**: Configure reverse proxy (Nginx/Traefik)

---

## 📊 Monitoring & Scaling

### Health Check Endpoint
Your app includes `/health` endpoint for monitoring:
```bash
curl https://your-app.herokuapp.com/health
```

### Scaling Options
- **Heroku**: `heroku ps:scale web=2`
- **Docker**: Use docker-compose with multiple replicas
- **VPS**: Setup load balancer + multiple Gunicorn workers

---

## 🎯 Recommended Hosting for Different Use Cases

| Use Case | Recommended Platform | Why |
|----------|---------------------|-----|
| **Personal/Testing** | Local + Heroku Free | Easy setup, no cost |
| **Small Team** | Railway/Render | Simple, automatic SSL |
| **Production** | Google Cloud Run | Scalable, reliable |
| **Enterprise** | VPS/Kubernetes | Maximum control |
| **Open Source** | GitHub Pages + API | Free hosting |

---

## 🔧 Troubleshooting

### Common Issues
1. **Port errors**: Make sure to use `PORT` environment variable
2. **Memory limits**: Reduce number of papers processed simultaneously
3. **Timeout issues**: Increase Gunicorn timeout settings
4. **API key issues**: Check environment variable names

### Logs
```bash
# Heroku
heroku logs --tail

# Local
python app.py  # Check console output

# VPS
sudo tail -f /var/log/research-system.log
```

---

## 🎉 Next Steps

1. **Choose your hosting option** based on your needs
2. **Test locally first** with `python app.py`
3. **Deploy to your chosen platform**
4. **Share your hosted research system** with others!

Your Multi-Agent Research System is now ready to help researchers worldwide! 🌍🔬 