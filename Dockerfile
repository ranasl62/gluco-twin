FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN pip install --no-cache-dir \
    numpy pandas scikit-learn matplotlib scipy tqdm \
    statsmodels seaborn openpyxl

COPY src/ /app/src/
COPY data/ /app/data/
COPY tools/ /app/tools/

ENV PYTHONPATH=/app
CMD ["python", "src/experiments/run_all.py"]
