# Getting Started
This project simulates the evolution of fluid flow and temperature distribution in a 2D domain over time. 
Specifically, it solves a set of partial differential equations (PDEs) that describe the fluid dynamics and heat transfer, 
typically associated with natural convection phenomena.

#### See [Explainer]("https://github.com/dhelafrank/Fluid-and-Temperature-Flow-Simulator/blob/master/EXPLAINER.md)

## 1. Installation
To get started make sure you have `Python (v3.5^)` and `Pip` Cofigure on your machine.

Next clone the repository: `git clone https://github.com/dhelafrank/Fluid-and-Temperature-Flow-Simulator.git`

## 2. Setup Virtual Enviroments 
It is neccesary to setup virtual enviroment for this project so the packages installed do not conflict with the global modules or packages.

### 2.0 (Linux)
#### Auto Setup: `sudo chmod +x ./linux-setup && ./linux-setup`
#### Manual Setup:
- Navigate to the project's root folder: `cd ./Fluid-and-Temperature-Flow-Simulator`
- Create virtual enviroment by executing the command: `python3 -m venv venv` where `venv` is the name of your virtual enviroment.
- Switch to virtual enviroment by executing: `source venv/bin/activate`

When you run `python` and or install packages and modules it will be executed from within the virtual enviroment.

- To leave virtual enviroment execute: `deactive`

### 2.1 (Windows)
- Install `virtualenv` by executing `pip install virtualenv`
- Check that virtualenv is installed through the command `pip --version`
- Install `virtualenvwrapper-win` by executing: `pip install virtualenvwrapper-win`
- Create a virtual enviroment by executing: `mkvirtualenv venv`. Replace `venv` with the project's name or whatever you chooses.
- This will create a folder named `Envs` in your user directory and set up the virtual environment inside it
- Navigate to the project root folder: `cd ./Fluid-and-Temperature-Flow-Simulator`.
- Set this folder as the project folder for your virtual enviroment: `setprojectdir .`
- Activate the virtual enviroment by executing: `workon venv`.

When you run `python` and or install packages and modules it will be executed from within the virtual enviroment.

- To leave virtual enviroment execute: `deactive`

## 3. Install project requirements
This project requires various modules and or packages to get it working.
- Make sure you are in the project's root folder
- Install requirements from requirements file by executing `pip install -r requirements.txt`
- Grab a coffe and wait for installtion to complete

## 4. Launch Project
- execute: `python3 main.py`
- follow the prompts
- find plots in `/Fluid-and-Temperature-Flow-Simulator/plots`