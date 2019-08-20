import glob
import os.path
import csv
import re

numb = re.compile(r"^(\w+)(\d+)$")

disambiguate_column = True
if disambiguate_column:
    disambiguation_own_column = ["Dis"]
    default_dis = ["_"]
else:
    disambiguation_own_column = []
    default_dis = []

for file in glob.glob("source/*.tsv"):
    name = os.path.basename(file)
    with open("tests/"+name, "w") as out:
        writer = csv.writer(out, delimiter="\t")
        with open(file) as inp:
            for ind, line in enumerate(csv.reader(inp, delimiter="\t")):

                if ind == 0:
                    writer.writerow(line+disambiguation_own_column)
                elif len(line) == 0:
                    writer.writerow(line)
                else:
                    token, lemma, pos, morph = line[0], line[1], line[2], line[3:]
                    dis = default_dis

                    if numb.match(lemma):
                        lemma, dis = numb.match(lemma).groups()
                        dis = [dis]
                    writer.writerow([token, lemma, pos] + morph + dis)
