import re


def transform_line(line):
    line = line.split('\t')

    doc = {
        'id': line[0],
        'label': line[1],
        'statement': line[2],
        'subject': line[3],
        'speaker': line[4],
        'job': line[5],
        'state': line[6],
        'party': line[7],
        'barely_true': line[8],
        'false': line[9],
        'half_true': line[10],
        'mostly_true': line[11],
        'pants_on_fire': line[12],
        'context': line[13]
    }

    statement = re.sub('[^A-Za-z\s]', '', doc['statement']).split(' ')
    return (statement, doc['label'])


def retrieve_docs(path):

    docs = []
    with open(path, 'r') as r:
        line = r.readline()

        while line:
            docs.append(transform_line(line))
            line = r.readline()

    return docs
