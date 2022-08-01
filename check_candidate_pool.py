import joblib
import glob

DATA_DIR = 'C:/Users/Simon/Desktop/prefetcher_outputs'


if __name__ == '__main__':
    job_files= glob.glob(f'{DATA_DIR}/*.joblib')
    print(f"Found {len(job_files)} joblib files")

    for file in job_files:
        data = joblib.load(file)
        print(data)