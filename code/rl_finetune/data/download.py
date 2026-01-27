from aoss_client.client import Client

client = Client()
url = "s3://jidong/train_tensor_dataset.pkl"
client.download_file(url,"./data/dataset_backups/train_tensor_dataset.pkl")
url = "s3://jidong/test_tensor_dataset.pkl"
client.download_file(url,"./data/dataset_backups/test_tensor_dataset.pkl")
url = "s3://jidong/edge.tar.gz"
client.download_file(url,"./edge.tar.gz")