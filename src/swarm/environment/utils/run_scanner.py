#!/usr/bin/env python3
"""
Launch script for the 2-dot quantum array scanners.
"""

import sys
import subprocess
import os

def main():
    print("🔬 2-Dot Quantum Array Scanner Launcher")
    print("=" * 50)
    print()
    print("Available scanners:")
    print("1. Mapping Scanner (qarray_simple_scanner.py)")  
    print("2. Real-time Scanner (qarray_realtime_scanner.py)")
    print()
    
    while True:
        try:
            choice = input("Enter your choice (1-2) or 'q' to quit: ").strip().lower()
            
            if choice == 'q':
                print("Goodbye!")
                return
            
            if choice == '1':
                script = 'qarray_simple_scanner.py'
                break
            elif choice == '2':
                script = 'qarray_realtime_scanner.py'
                break
            else:
                print("Invalid choice. Please enter 1, 2, or 'q'.")
                continue
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            return
    
    print(f"\nLaunching {script}...")
    print("The web interface will open in your browser automatically.")
    print("Press Ctrl+C to stop the server.")
    print("-" * 50)
    
    # Launch the selected scanner
    try:
        current_dir = os.path.dirname(__file__)
        script_path = os.path.join(current_dir, script)
        
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", script_path,
            "--server.port", "8501",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ])
        
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except Exception as e:
        print(f"Error launching scanner: {e}")

if __name__ == "__main__":
    main()