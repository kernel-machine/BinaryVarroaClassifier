## HOW TO RUN
This software is build to run with Docker, but your are free to run the code also without the container.
### Prepare the container (necessary only if you use container)

- in `.devcontainer/.devcontainer.json` edit the path of the mounted folder with the one of dataset
- If you are using VSCode, press `Ctrl+Maiusc+P` and then, "build and open in a container "

### Extract the dataset
Extract the .7z file in the mounted path
### Run the evaluation code
- Create a virtual environment with python 
```bash
python3.11 -m venv env
```
- Install all the python dependecies
```bash
./env/bin/pip install -r requirements.txt
```
- Run the code to evaluate the model
```bash
cd src
../env/bin/python evaluate.py --input_dataset /dataset --model model.pth
```
The path of the dataset is the folder containing the images