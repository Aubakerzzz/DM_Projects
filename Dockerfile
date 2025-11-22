FROM python:3.10

# Avoid Python buffering
ENV PYTHONUNBUFFERED=1

# Set workdir
WORKDIR /app

# Copy requirements first to leverage Docker caching
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the notebook and all code
COPY . .

# Expose Jupyter port
EXPOSE 8888

# Start Jupyter Lab when container runs
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
