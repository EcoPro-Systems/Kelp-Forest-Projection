# Use an official Miniconda base image
FROM continuumio/miniconda3

# Set working directory in the container
WORKDIR /app

# Install git
RUN apt-get update && apt-get install -y git

# Clone the repository
RUN git clone https://github.com/EcoPro-Systems/Kelp-Forest-Projection .

# Create the conda environment
RUN conda create -n kelp python=3.10 -y

# Install packages using conda run
RUN conda run -n kelp conda install -y ipython jupyter pandas matplotlib scipy scikit-learn && \
    conda run -n kelp conda install -y -c conda-forge xarray dask netCDF4 bottleneck && \
    conda run -n kelp pip install tqdm statsmodels astropy

# Set the default command to activate the conda environment and start a bash shell
SHELL ["conda", "run", "-n", "kelp", "/bin/bash", "-c"]
CMD ["/bin/bash"]