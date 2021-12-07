# Uniform-Control-Cellular-Robot
## Required packages and environments
The code is tested with Python3.9.7.
Required packages: numpy matplotlib torch

Current pytorch version: 1.10.0

Nvidia Driver Version: 470.86       

CUDA Version: 11.4   
## To run the code:
Open the terminal from the project folder:
### To train the network: 
Run it in the terminal: `python main.py --train_dqn`

### To test the network:  
After training, in '.\parameters' folder, all network parameter is saved as '.pth' file, choose the parameter you want to use and put it in the project folder and rename it as 'DQNmodelgpu.pth'.

Run it in the terminal: `python main.py --test_dqn`

If you get an error while running “python main.py --train_dqn” like:
`ModuleNotFoundError：No module named ‘tkinter’`

Please run command in terminal： `sudo apt-get install python3-tk`
This module is for realtime plotting in matplotlib using 'TkAgg'.
