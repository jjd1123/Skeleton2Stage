from aoss_client.client import Client
client = Client()
# url = "s3://jidong/train_tensor_dataset.pkl"
# client.upload_file(url,"./dataset_backups/train_tensor_dataset.pkl")
# url = "s3://jidong/test_tensor_dataset.pkl"
# client.upload_file(url,"./dataset_backups/test_tensor_dataset.pkl")
# url = "s3://jidong/edge.tar.gz"
# client.upload_file(url,"/mnt/Nerf001/jidong/edge.tar.gz")
url = "s3://jidong/data.tar"
client.upload_file(url,"/mnt/Nerf001/jidong/EDGE/data/data.tar")


