FROM neungkl/keras

RUN pip install Flask
RUN pip install werkzeug
RUN apt-get update
COPY . .

CMD ["sh", "server.sh"]