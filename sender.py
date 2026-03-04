# %%

import argparse
import os

import dotenv
import boto3
from tqdm import tqdm 

dotenv.load_dotenv()

AWS_KEY = os.getenv("AWS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")

class Sender:

    def __init__(self, bucket_name, bucket_folder):
        self.bucket_name = bucket_name
        self.bucket_folder = bucket_folder
        
        self.s3 = boto3.client("s3",
                        aws_access_key_id=AWS_KEY,
                        aws_secret_access_key=AWS_SECRET_KEY,
                        region_name="us-east-1")


    def process_file(self, filename):

        file = filename.split("/")[-1]
        bucket_path = os.path.join(self.bucket_folder, file)

        try:
            self.s3.upload_file(filename,
                                self.bucket_name,
                                bucket_path)
            
        except Exception as err:
            print(err)
            return False

        os.remove(filename)
        return True


    def process_folder(self, folder):
        files = [i for i in os.listdir(folder) if i.endswith(".parquet")]
        for f in tqdm(files):
            self.process_file(os.path.join(folder, f))


# %%

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", type=str)
    parser.add_argument("--bucket_path",default="f1/results", type=str)
    parser.add_argument("--folder", default="data", type=str)
    args = parser.parse_args()

    if args.bucket:
        send = Sender(args.bucket, args.bucket_path)
        send.process_folder(args.folder)

    else:
        print("sem bucket definido")