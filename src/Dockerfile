FROM python:3.7.5 as Trainer
COPY . .
RUN pip install -r requirements.txt && python main.py

FROM python:3.7.5
COPY /app /app
WORKDIR /app
RUN pip install -r requirements.txt
COPY --from=Trainer Model.pkl .
COPY --from=Trainer output_file.json .
CMD [ "python", "app.py" ]