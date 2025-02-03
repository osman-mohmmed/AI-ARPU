# Base image
FROM python:3.9

# Set the working directory in the container
# WORKDIR /app

# Copy the requirements file into the container
# COPY requirements.txt .

COPY apply_annSolution.py /codeBase/
COPY apply_olsSolution.py /codeBase/
COPY ScrapingandPreprocessing.py /codeBase/
COPY createANNModel.py /codeBase/
COPY LinearRegDiagnostic.py /codeBase/

# Copy only the requirements file first (so this layer is cached if unchanged)
COPY requirements.txt .

# Install dependencies (cached unless requirements.txt changes)
RUN pip install --no-cache-dir -r requirements.txt

# Install Python libraries
# RUN pip install --no-cache-dir  numpy
# RUN pip install --no-cache-dir  pandas
# RUN pip install --no-cache-dir  tensorflow
# RUN pip install --no-cache-dir requests 
# RUN pip install --no-cache-dir bs4
# RUN pip install --no-cache-dir tabulate
# RUN pip install --no-cache-dir seaborn
# RUN pip install --no-cache-dir scikit-learn
# RUN pip install --no-cache-dir scipy
# RUN pip install --no-cache-dir statsmodels
# RUN pip install --no-cache-dir matplotlib

# # Add README
COPY README.md /codeBase/
 
# Metadata
LABEL maintainer="Sara & Osman"
LABEL course="M.Grum: Advanced AI-based Application Systems"
LABEL university="University of Potsdam"
LABEL license="AGPL-3.0"

CMD ["sh", "-c", "python3"]  # Use sh instead of bash