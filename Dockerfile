FROM python:3.10

WORKDIR /app

# Copy project files into /app
COPY . /app

# Upgrade pip
RUN pip install --upgrade pip

# Install dependencies
RUN pip install -r requirements.txt

# Run Streamlit (Render-compatible)
CMD ["python", "-m", "streamlit", "run", "/web-app-new.py", "--server.port=$PORT", "--server.address=0.0.0.0"]