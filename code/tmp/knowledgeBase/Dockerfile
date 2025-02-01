# Base image
FROM busybox
 
# Create directories for data
# RUN mkdir -p /tmp/knowledgeBase
 
# Add model
ADD ANN_model.keras /knowledgeBase/

 
# Add README
ADD README.md /knowledgeBase/

 
# Metadata
# LABEL maintainer="Sara & Osman"
# LABEL course="M.Grum: Advanced AI-based Application Systems"
# LABEL university="University of Potsdam"
# LABEL license="AGPL-3.0"

# CMD ["sh", "-c"]  # Use sh instead of bash