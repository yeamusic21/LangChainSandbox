from config.tp_secrets import Secrets

if __name__=="__main__":
    os.environ['INDEX_NAME'] = Secrets.pinecode_index_name
    os.environ['PINECODE_API_KEY'] = Secrets.pinecode_api_key
    print("Ingesting ... ")