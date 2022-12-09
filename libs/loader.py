import main
from termcolor import colored
import os

version = "2.5.0"

os.system('cls' if os.name == 'nt' else 'clear')

print(colored(f"""
     :::.   :::::::-.    :::.     ::::::
     ;;`;;   ;;,   `';,  ;;`;;    ;;;;;;
    ,[[ '[[, `[[     [[ ,[[ '[[,  [[[[[[
   c$$$cc$$$c $$,    $$c$$$cc$$$c $$$$$$
    888   888,888_,o8P' 888   888,888888
    YMM   ""` MMMMP"`   YMM   ""` MMMMMM   *{version}
     
       (Neural Network Aimbot)
""", "yellow"))

main.start(onEnable=True) #False for spectator mode
