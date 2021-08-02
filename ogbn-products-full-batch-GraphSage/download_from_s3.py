# download_from_s3.py
# python download_from_s3.py -b "ogb-products" -o "./dataset/" -p ogbn_products -f ogbn_products_dgl
import os
import shutil
import boto3
import time
import argparse

def get_s3_bucket_object_list(bucket_name="ogb-products"):
    s3_bucket = boto3.resource('s3').Bucket(bucket_name)
    keys = [obj.key for obj in s3_bucket.objects.all()]
    return keys

def make_intermediate_directory_structure(s3_object_keys = [], output_path="./dataset-test/"):
    for key in s3_object_keys:
        split = key.split('/')
        for i, parent in enumerate(split[:-1]):
            dir = str()
            for j in range(i):
                dir += split[j] + "/"
            try: 
                os.mkdir(output_path+dir+parent)
            except FileNotFoundError:
                try:
                    os.mkdir(output_path)
                    os.mkdir(output_path+dir+parent)
                except:
                    print("not working with ogbn_products?")
            except FileExistsError:
                pass

def download_s3_bucket_object_list(bucket_name="ogb-products", 
                                   s3_object_keys=[],
                                   output_path="./dataset-test/"):
    [os.system(f"") for key in s3_object_keys]
    s3_bucket = boto3.resource('s3').Bucket(bucket_name)
    [s3_bucket.download_file(key, f"{output_path}/{key}") for key in s3_object_keys]

def change_subdirectory_name(previous_subdirectory_name, final_subdirectory_name):
    shutil.move(previous_subdirectory_name, final_subdirectory_name)

def get_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--bucket_name")
    parser.add_argument("-o", "--output_path")
    parser.add_argument("-p", "--previous_subdirectory_name") # ogbn_products
    parser.add_argument("-f", "--final_subdirectory_name")    # ogbn_products_dgl
    arg_values = parser.parse_args()
    return arg_values 

if __name__ == "__main__":
    print("\n\nDOWNLOADING DATA FROM S3", end="\n")
    arg_values = get_cli_args()    
    t0 = time.time()
    s3_object_keys = get_s3_bucket_object_list(arg_values.bucket_name)
    if os.path.exists(arg_values.output_path):
        shutil.rmtree(arg_values.output_path)
    make_intermediate_directory_structure(s3_object_keys=s3_object_keys, output_path=arg_values.output_path)
    download_s3_bucket_object_list(s3_object_keys=s3_object_keys, output_path=arg_values.output_path)
    change_subdirectory_name(os.path.join(arg_values.output_path, arg_values.previous_subdirectory_name), 
                             os.path.join(arg_values.output_path, arg_values.final_subdirectory_name))
    tf = time.time()
    print(f"Data download time: {round(tf-t0, 3)} seconds", end="\n\n")