FROM python:3.12
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
EXPOSE 80
CMD streamlit run app.py --server.port 8080 --server.address 0.0.0.0 