# ---------------------------
# Stage 1: Build the React Frontend
# ---------------------------
    FROM node:18-alpine AS frontend-builder

    # Set the working directory to /app
    WORKDIR /app
    
    # Copy React package files to install dependencies first
    COPY package*.json ./
    
    # Install dependencies for the React app
    RUN npm install --legacy-peer-deps
    
    # Copy the rest of the React source code (public and src folders)
    COPY public/ public/
    COPY src/ src/
    
    # Build the React app (this creates a "build" folder with production files)
    RUN npm run build
    
    # ---------------------------
    # Stage 2: Set Up the Flask Backend
    # ---------------------------
    FROM python:3.9-slim
    
    # Set the working directory
    WORKDIR /app
    
    # Copy the Flask backend code into the container
    COPY backend/ ./backend
    
    # Copy the built React app from the first stage into your Flask app's static folder.
    # (Make sure your Flask app is set up to serve static files from, for example, backend/static)
    COPY --from=frontend-builder /app/build ./backend/static
    
    # Change to the backend directory
    WORKDIR /app/backend
    
    # Copy and install Python dependencies for your Flask app
    COPY backend/requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt
    
    # Expose the port (Cloud Run expects 8080 by default)
    EXPOSE 8080
    ENV PORT 8080
    
    # Start your Flask app using Gunicorn.
    # This command assumes your Flask app is defined in a file (e.g., app.py) with an app instance named "app".
    CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
    