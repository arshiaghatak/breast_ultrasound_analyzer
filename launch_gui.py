#!/usr/bin/env python3
"""
Simple launcher script for the Breast Cancer Classifier GUI
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from breast_cancer_gui import main
    
    print("Starting Breast Cancer Classifier GUI...")
    print("=" * 50)
    main()
    
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please make sure all required packages are installed:")
    print("pip install -r requirements.txt")
    sys.exit(1)
    
except Exception as e:
    print(f"Error starting GUI: {e}")
    sys.exit(1)
