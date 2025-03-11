## Demo with jupyter

The notebook in the base repo demo.ipynb will walk through the 4 tasks described and output my process/thoughts along the way with some of the code. Most of this more front end-ish code is in showcase.py and main.py, which both simply serve as the "glue" to demo capabilities of the main project code. In addition, intial_setup.py also is responsible for generating a lot of test/demo data. For an understanding of the codebase, focus on the actual subdirectories data, model and training for the actual core functionality.

This project is managed with poetry.

Download and add requirements

poetry install

poetry run jupyter notebook

Then double click the demo.ipynb file to go through the demonstration.

## Training Mode in Docker

This demo includes a dockerized version of a quick "training" mode for 10 epochs with made up data.

docker build -f Dockerfile -t sentence-transformer-demo .

docker run -p 8888:8888 -v $(pwd):/workspace sentence-transformer-demo

On Windows,

docker run -p 8888:8888 -v %cd%:/workspace sentence-transformer-demo

There is also a serving docker that I have not had chance to test/get to yet beyond setting up the basics.