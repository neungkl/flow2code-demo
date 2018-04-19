FROM neungkl/keras

RUN pip install Flask
RUN pip install werkzeug
RUN apt-get update

CMD ["FLASK_APP=main.py", "flask", "run"]