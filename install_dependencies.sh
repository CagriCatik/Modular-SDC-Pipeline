#!/bin/bash

# Bash Script to Install Dependencies for Manual Car Driving Script
# Author: [Your Name]
# Date: $(date +"%Y-%m-%d")

# Exit immediately if a command exits with a non-zero status
set -e

echo "Starting installation of packages..."

# Update and upgrade system packages
echo "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install system dependencies
echo "Installing system dependencies..."
sudo apt-get install -y python3-dev python3-pip python3-venv
sudo apt-get install -y libgl1-mesa-glx libglu1-mesa
sudo apt-get install -y swig
sudo apt-get install -y xvfb  # Virtual framebuffer (optional)

# Create a virtual environment (optional but recommended)
echo "Creating a virtual environment..."
python3 -m venv venv

# Activate the virtual environment
echo "Activating the virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install Python packages
echo "Installing Python packages..."
pip install --upgrade numpy
pip install --upgrade gym[box2d]
pip install --upgrade pyglet
pip install --upgrade box2d-py
pip install --upgrade pygame

# Deactivate the virtual environment
echo "Deactivating the virtual environment..."
deactivate

echo "Installation complete!"

echo ""
echo "To activate the virtual environment, run:"
echo "source venv/bin/activate"
echo ""
echo "Then, to run your script, execute:"
echo "python manual_car_driving.py"

