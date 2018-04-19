FROM neungkl/keras

RUN pip install Flask
RUN pip install werkzeug

CMD ["FLASK_APP=main.py", "flask", "run"]