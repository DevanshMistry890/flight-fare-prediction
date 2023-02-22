FROM continuumio/anaconda3
COPY . /usr/app/
EXPOSE 5000
WORKDIR /usr/app/
RUN pip install -r requirements.txt
ENTRYPOINT [ "python" ]
CMD ["app.py"]