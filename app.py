import subprocess
import os
from concurrent.futures import ProcessPoolExecutor

# Function to run a Streamlit script using subprocess
def run_streamlit_script(script_path):
    subprocess.run(["streamlit", "run", script_path])

if __name__ == "__main__":
    
    # Define the main function
    def main():
        # Get the directory of the current script
        app_directory = os.path.dirname(os.path.abspath(__file__))
        
        # Define the path to the Streamlit app script
        model_path = os.path.join(app_directory, "app", "model.py")

        # Execute the Streamlit script using ProcessPoolExecutor
        with ProcessPoolExecutor() as executor:
            executor.submit(run_streamlit_script, model_path)

    # Check if the script is being run directly
    if __name__ == "__main__":
        # Call the main function
        main()
