# Base image
FROM busybox
 
# Create directories for data
RUN mkdir -p /tmp/codeBase
 
# Copy data files
COPY ../../../code/src.py /tmp/codeBase/

 
# Add README
COPY README.md /tmp/codeBase/
 
# Metadata
LABEL maintainer="Sara & Osman"
LABEL course="M.Grum: Advanced AI-based Application Systems"
LABEL university="University of Potsdam"
LABEL license="AGPL-3.0"

CMD ["sh", "-c"]  # Use sh instead of bash