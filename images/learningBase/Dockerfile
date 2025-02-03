# Base image
FROM busybox
 
# Create directories for data
# RUN mkdir -p /tmp/learningBase/train /tmp/learningBase/validation
 
# Copy data files
COPY ../../../data/training_data.csv /learningBase/train/
COPY ../../../data/test_data.csv /learningBase/validation/
 
# Add README
COPY README.md /learningBase/
 
# Metadata
LABEL maintainer="Sara & Osman"
LABEL course="M.Grum: Advanced AI-based Application Systems"
LABEL university="University of Potsdam"
LABEL license="AGPL-3.0"

CMD ["sh", "-c"]  # Use sh instead of bash

