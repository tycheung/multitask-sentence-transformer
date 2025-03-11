FROM tensorflow/serving:2.10.0

COPY ./models /models

COPY ./serving/models.config /models/models.config

ENV MODEL_CONFIG_FILE=/models/models.config
ENV MODEL_NAME=sentence_transformer

EXPOSE 8500
EXPOSE 8501

CMD ["tensorflow_model_server", "--port=8500", "--rest_api_port=8501", "--model_config_file=${MODEL_CONFIG_FILE}"]