@echo off
echo Starting Sentence Transformer application with Jupyter notebook

mkdir data 2>nul
mkdir models 2>nul

docker-compose up -d

echo.
echo Services started!
echo Jupyter notebook is available at: http://localhost:8888
echo TensorFlow Serving is available at: http://localhost:8501
echo.
echo To view the logs, run: docker-compose logs -f
echo To stop the services, run: docker-compose down