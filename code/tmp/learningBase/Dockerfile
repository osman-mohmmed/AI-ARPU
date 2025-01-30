# Base image
FROM busybox
 
# Create directories for data
# RUN mkdir -p /tmp/learningBase/train /tmp/learningBase/validation
 
# Copy data files
COPY ../../../data/training_data.csv /tmp/learningBase/train/
COPY ../../../data/test_data.csv /tmp/learningBase/validation/
 
# Add README
COPY README.md /tmp/learningBase/
 
# Metadata
LABEL maintainer="Sara & Osman"
LABEL course="M.Grum: Advanced AI-based Application Systems"
LABEL university="University of Potsdam"
LABEL license="AGPL-3.0"