import glob
import os.path
import os
import csv
import re

numb = re.compile(r"^(\w+)(\d+)$")


def make_parser(*args, instantiator, **kwargs):
    parser = instantiator(*args, description="Transform a Pie-Tab corpus for Tarte",
                          help="Transform a Pie-Tab corpus for Tarte",
                                     **kwargs)

    parser.add_argument("output_dir", help="If set, directory where data should be saved", type=str)
    parser.add_argument("files", nargs="+", help="Files that should be dispatched for Tarte", type=str)
    return parser


def main(args):
    # Create dir
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    # Parse the arguments
    disambiguate_column = True
    if disambiguate_column:
        disambiguation_own_column = ["Dis"]
        default_dis = ["_"]
    else:
        disambiguation_own_column = []
        default_dis = []

    for file in args.files:
        name = os.path.basename(file)
        with open(os.path.join(args.output_dir, name), "w") as out:
            writer = csv.writer(out, delimiter="\t")
            with open(file) as inp:
                for ind, line in enumerate(csv.reader(inp, delimiter="\t")):

                    if ind == 0:
                        if len(line) == 0:
                            print("You need to write headers manually in the output file !")
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
