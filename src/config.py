import os
import dotenv

project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
dotenv_path = os.path.join(project_dir, '.env')
dotenv.load_dotenv(dotenv_path)

# datasets
data_path = os.path.join(project_dir, 'data')
data_raw_path = os.path.join(data_path, 'raw')
data_proc_path = os.path.join(data_path, 'processed')