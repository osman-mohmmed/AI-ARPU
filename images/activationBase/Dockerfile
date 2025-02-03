# Base image
FROM busybox
 
# Create directories for data
# RUN mkdir -p /tmp/activationBase
 
# Copy data files
COPY activation_data.csv /activationBase/

 
# Add README
COPY README.md /activationBase/
 
# Metadata
LABEL maintainer="Sara & Osman"
LABEL course="M.Grum: Advanced AI-based Application Systems"
LABEL university="University of Potsdam"
LABEL license="AGPL-3.0"

CMD ["sh", "-c"]  # Use sh instead of bash