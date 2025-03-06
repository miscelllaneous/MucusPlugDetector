FROM nvcr.io/nvidia/pytorch:24.08-py3

RUN apt-get update && apt-get install -y 

RUN pip install \
    SimpleITK==2.4.0 \
    nibabel==5.3.2 \
    scikit-image==0.22.0 \
    monai==1.4.0 \
    pytorch-lightning==2.4.0 \
    lungmask==0.2.20

ENTRYPOINT ["/bin/bash"]