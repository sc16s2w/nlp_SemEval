import csv

def write_to_tsv(output_path: str, file_columns: list, data: list):
    csv.register_dialect('tsv_dialect', delimiter='\t', quoting=csv.QUOTE_ALL)
    with open(output_path, "w", newline="") as wf:
        writer = csv.DictWriter(wf, fieldnames=file_columns, dialect='tsv_dialect')
        writer.writerows(data)
    csv.unregister_dialect('tsv_dialect')

def read_from_tsv(file_path: str, column_names: list) -> list:
    csv.register_dialect('tsv_dialect', delimiter='\t', quoting=csv.QUOTE_ALL)
    with open(file_path, "r") as wf:
        reader = csv.DictReader(wf, fieldnames=column_names, dialect='tsv_dialect')
        datas = []
        for row in reader:
            data = dict(row)
            datas.append(data)
    csv.unregister_dialect('tsv_dialect')
    return datas

def create_test():
    with open("dataset/testset-levelc.tsv", "r") as test, open("dataset/labels-levelc.csv", "r") as label, open(
            "dataset/test_task3.txt", "w+") as file:

        for a, b in zip(test, label):
            print(a,b)
            tokens = b.split(",")
            test_tokens = a.split("\t")
            assert test_tokens[0] == tokens[0]
            file.write(a[:-1] + "\t" + tokens[1])

def create_dev():
    data = read_from_tsv("/Users/wangsiwei/PycharmProjects/pythonProject2/dataset/olid-training-v1.0.tsv",
                         ["id", "tweet", "subtask_a", "subtask_b", "subtask_c"])
    data_train = data[0:12241]
    data_validation = [data[0]]
    data_validation.extend(data[12241:13240])
    write_to_tsv("/Users/wangsiwei/PycharmProjects/pythonProject2/dataset/train.tsv",
                 ["id", "tweet", "subtask_a", "subtask_b", "subtask_c"], data_train)
    write_to_tsv("/Users/wangsiwei/PycharmProjects/pythonProject2/dataset/dev.tsv",
                 ["id", "tweet", "subtask_a", "subtask_b", "subtask_c"], data_validation)

##python bert.py --data_dir data/ --output_dir output/ --do_test --do_lower_case --bert_model bert-base-uncased

## python main.py --task_name one --do_train --do_test --do_lower_case --data_dir ./dataset/ --output_dir ./try_task1/ --bert_model bert-base-uncased --max_seq_length 80 --train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 2.0


if __name__ == "__main__":
    create_test()

