# Use official Python image (Full version, not slim) to avoid missing library errors
FROM python:3.9

# Allow statements and log messages to immediately appear in the Knative/Railway logs
ENV PYTHONUNBUFFERED True

# Copy local code to the container image
ENV APP_HOME /app
WORKDIR $APP_HOME

# Install production dependencies
COPY requirements.txt ./
# timeout increased for heavy tensorflow install
RUN pip install --no-cache-dir -r requirements.txt --timeout 1000

# Copy local code to the container image
COPY . ./

# Expose port
ENV PORT 8080

# Run with Gunicorn
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
