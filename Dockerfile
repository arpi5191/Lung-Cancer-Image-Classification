FROM python:3.10

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    libhdf5-dev \
    libgl1-mesa-dev \
    libglib2.0-0 \
    build-essential \
    libffi-dev \
    libssl-dev \
    zlib1g-dev

# Upgrade pip, setuptools, and wheel
RUN pip install --upgrade pip setuptools wheel

# Install specific packages
RUN pip install --no-binary=h5py h5py stardist tensorflow scikit-image tifffile numpy argparse opencv-python
RUN mkdir python

COPY seg.py python/

# larger cancer cell default min.
# ENV NMIN=10000
# smaller healthy cell default min
ENV NMIN=2000
ENV NMAX=1000000
ENV DIDX=0
# stardist likely trained on images 4x less zoomed in than expansion
ENV DOWNSAMPLE=4

CMD python python/seg.py --nmin $NMIN --nmax $NMAX --didx $DIDX --d $DOWNSAMPLE
