# ---------------------------
    # Stage 1: Build the React Frontend
    # ---------------------------
        FROM node:18-alpine AS frontend-builder

        WORKDIR /app
        COPY package*.json ./
        RUN npm install --legacy-peer-deps
        COPY public/ public/
        COPY src/ src/
        RUN npm run build
    
        # ---------------------------
        # Stage 2: Set Up the Flask Backend
        # ---------------------------
        FROM python:3.9-slim
    
        # Keep the working directory as /app
        WORKDIR /app
    
        # Copy the Flask backend code
        COPY backend/ ./backend
    
        # Copy the built React app
        COPY --from=frontend-builder /app/build ./backend/static
    
        # Copy requirements.txt (relative to /app)
        COPY backend/requirements.txt ./backend/requirements.txt
    
        # Install Python dependencies
        WORKDIR /app/backend  # <--- Change to /app/backend
        RUN pip install --no-cache-dir -r requirements.txt
    
        WORKDIR /app # <--- Change back to /app for Gunicorn
    
        EXPOSE 8080
        ENV PORT 8080
    
        CMD ["gunicorn", "--bind", "0.0.0.0:8080", "backend.app:create_app()"] # <--- Correct path