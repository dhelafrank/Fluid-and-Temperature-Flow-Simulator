# Getting Started
This project simulates the evolution of fluid flow and temperature distribution in a 2D domain over time. 
Specifically, it solves a set of partial differential equations (PDEs) that describe the fluid dynamics and heat transfer, 
typically associated with natural convection phenomena.
## 1 Setup Virtual Enviroments 
It is neccesary to setup virtual enviroment for this project so the packages installed do not conflict with the global modules or packages.

### 1.0 (Linux)
#### Auto Setup: `sudo chmod +x ./linux-setup && ./linux-setup`
#### Manual Setup:
- Navigate to the project's root folder: `cd ./core`
- Create virtual enviroment by executing the command: `python3 -m venv obf` where `obf` is the name of your virtual enviroment.
- Switch to virtual enviroment by executing: `source obf/bin/activate`

When you run `python` it will be executed from within this virtual enviroment.

- To leave virtual enviroment execute: `deactive`

### 1.1 (Windows)
- Install `virtualenv` by executing `pip install virtualenv`
- Check that virtualenv is installed through the command `pip --version`
- Install `virtualenvwrapper-win` by executing: `pip install virtualenvwrapper-win`
- Create a virtual enviroment by executing: `mkvirtualenv obf`. Replace `obf` with your project's name.
- This will create a folder named `Envs` in your user directory and set up the virtual environment inside it
- Navigate to the project root folder: `cd ./core`.
- Set this folder as the project folder for your virtual enviroment: `setprojectdir .`
- Activate the virtual enviroment by executing: `workon obf`.

When you run `python` and or install packages and modules it will be executed from within the virtual enviroment.

- To leave virtual enviroment execute: `deactive`

## 2. Install project requirements
This project requires various modules and or packages to get it working.
- Make sure you are in the project's root folder
- Install requirements from requirements file by executing `pip install -r requirements.txt`
- Grab a coffe and wait for installtion to complete

## 3. Launch Project