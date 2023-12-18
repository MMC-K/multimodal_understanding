import os
import csv
import re
import utils


FASHION_TEST_QUERY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/fashion_test_queries.csv")

all_tags = set()

for tag in utils.id_to_tags_dict.values():
    all_tags = all_tags.union(tag)

with open(FASHION_TEST_QUERY_PATH, newline='\n') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    next(csv_reader, None)
    print("[*] ")
    for row in csv_reader:
        natural_language_description = row[1]
        included_tags = {t.strip() for t in re.split('\n|,', row[2]) if len(t.strip()) > 0}
        clothes_tags = row[3]
        type = row[4]
        url = row[5]

        print(natural_language_description, included_tags, all_tags, type, url)

        # img_id = url.split("/")[-1].split(".")[0]
        # db_tags = utils.id_to_tags(img_id)

        for t in included_tags:
            print(t)
            if t not in all_tags:
                input("[*] Invalid tag found!:", t)

print("[*] Validation finished")