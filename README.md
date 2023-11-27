# CS-357
This repository contains the code developed for the minor project made as part of the CS 357 Optimization Algorithm and Techniques Lab. The topic of the project is Robust Planning of Airport Platform Buses.

### Tech Stack Used 
1. Python: utilized for implementing ILP models
2. JSON: employed for storing and handling input data

### Project Hierarchy
The codebase is organized into the following components:
- `bus_schedule.py`: Contains the main code for the project where the column generation technique is implemented
- `data.json`: Generated for the input data when data.py is executed
- `data.py`: Used to store the input data
- `data_loader.py`: Contains function to load the JSON file

### Pre-requisites
Install the Gurobi solver for optimization and obtain the corresponding license.
Create a virtual environment and install the dependencies.
```bash
$ python -m venv /path/to/new/virtual/environment
$ pip install -r requirements.txt
```

### Implementation
The code can be implemented as follows:
```bash
# To create the data.json file
$ python data.py
# To implement the main code
$ python bus_schedule.py
```

### Contributors
- Agrima Bundela (210002009)
- Rishika Sharma (210002063)
- Niranjana R Nair (210003049)

### Acknowledgement
We express our gratitude to Dr. Kapil Ahuja, Professor, Department of Computer Science and Engineering, IIT Indore. We have gained valuable insights through the completion of this project for the associated lab course under his guidance.

