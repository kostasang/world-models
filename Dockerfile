FROM duckietown/gym-duckietown

WORKDIR /workspace
COPY scripts /workspace/scripts
COPY world_models /workspace/world_models
COPY configs /workspace/configs
COPY requirements.in /workspace/
COPY setup.py /workspace/

RUN ls world_models
RUN pip install --upgrade pip && \
    pip3 install -e .
