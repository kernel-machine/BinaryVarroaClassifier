## HOW TO RUN

This software is conterized with Podman, (you can also used Docker)

- Create a virtual environment with python 
```bash
python -m venv env
```
- Install all the python dependecies
```bash
./env/bin/pip install -r requirements.txt
```
- Run the code to evaluate the model
```bash
source env/bin/activate
cd src
python evaluate.py --input_dataset /dataset/varroa_visible_box --model model.pth
```