
import os
import csv
import gzip
import collections
import re
import io
import json
import xml.etree.ElementTree as ET

import requests
import pandas


def Test2():
    xml_path = os.path.join('download', 'drugbank.xml.gz')
    #print(xml_path)
    xml_path="C:/git/DMPrj/src/download/drugbank.xml.gz"
    with gzip.open(xml_path) as xml_file:
        tree = ET.parse(xml_file)
    root = tree.getroot()
    return ("root")


def DrugbankDataSets():

    xml_path = os.path.join('download', 'drugbank.xml.gz')
    #print(xml_path)
    xml_path="C:/git/DMPrj/src/download/drugbank.xml.gz"
    with gzip.open(xml_path) as xml_file:
        tree = ET.parse(xml_file)
    root = tree.getroot()

    ns = '{http://www.drugbank.ca}'
    inchikey_template = "{ns}calculated-properties/{ns}property[{ns}kind='InChIKey']/{ns}value"
    inchi_template = "{ns}calculated-properties/{ns}property[{ns}kind='InChI']/{ns}value"

    rows = list()
    for i, drug in enumerate(root):
        row = collections.OrderedDict()
        assert drug.tag == ns + 'drug'
        row['type'] = drug.get('type')
        row['drugbank_id'] = drug.findtext(ns + "drugbank-id[@primary='true']")
        row['name'] = drug.findtext(ns + "name")
        row['description'] = drug.findtext(ns + "description")
        row['groups'] = [group.text for group in
                         drug.findall("{ns}groups/{ns}group".format(ns=ns))]
        row['atc_codes'] = [code.get('code') for code in
                            drug.findall("{ns}atc-codes/{ns}atc-code".format(ns=ns))]
        row['categories'] = [x.findtext(ns + 'category') for x in
                             drug.findall("{ns}categories/{ns}category".format(ns=ns))]
        row['inchi'] = drug.findtext(inchi_template.format(ns=ns))
        row['inchikey'] = drug.findtext(inchikey_template.format(ns=ns))

        # Add drug aliases
        aliases = {
            elem.text for elem in
            drug.findall("{ns}international-brands/{ns}international-brand".format(ns=ns)) +
            drug.findall("{ns}synonyms/{ns}synonym[@language='English']".format(ns=ns)) +
            drug.findall("{ns}international-brands/{ns}international-brand".format(ns=ns)) +
            drug.findall("{ns}products/{ns}product/{ns}name".format(ns=ns))

        }
        aliases.add(row['name'])
        row['aliases'] = sorted(aliases)

        rows.append(row)

    alias_dict = {row['drugbank_id']: row['aliases'] for row in rows}
    with open('C:/git/DMPrj/src/download/data/aliases.json', 'w') as fp:
        json.dump(alias_dict, fp, indent=2, sort_keys=True)

    def collapse_list_values(row):
        for key, value in row.items():
            if isinstance(value, list):
                row[key] = '|'.join(value)
        return row

    rows = list(map(collapse_list_values, rows))

    columns = ['drugbank_id', 'name', 'type', 'groups', 'atc_codes', 'categories', 'inchikey', 'inchi', 'description']
    drugbank_df = pandas.DataFrame.from_dict(rows)[columns]

    drugbank_slim_df = drugbank_df[
        drugbank_df.groups.map(lambda x: 'approved' in x) &
        drugbank_df.inchi.map(lambda x: x is not None) &
        drugbank_df.type.map(lambda x: x == 'small molecule')
    ]


    protein_rows = list()
    for i, drug in enumerate(root):
        drugbank_id = drug.findtext(ns + "drugbank-id[@primary='true']")
        for category in ['target', 'enzyme', 'carrier', 'transporter']:
            proteins = drug.findall('{ns}{cat}s/{ns}{cat}'.format(ns=ns, cat=category))
            for protein in proteins:
                row = {'drugbank_id': drugbank_id, 'category': category}
                row['organism'] = protein.findtext('{}organism'.format(ns))
                row['known_action'] = protein.findtext('{}known-action'.format(ns))
                actions = protein.findall('{ns}actions/{ns}action'.format(ns=ns))
                row['actions'] = '|'.join(action.text for action in actions)
                uniprot_ids = [polypep.text for polypep in protein.findall(
                    "{ns}polypeptide/{ns}external-identifiers/{ns}external-identifier[{ns}resource='UniProtKB']/{ns}identifier".format(ns=ns))]
                if len(uniprot_ids) != 1:
                    continue
                row['uniprot_id'] = uniprot_ids[0]
                ref_text = protein.findtext("{ns}references[@format='textile']".format(ns=ns))
                pmids = re.findall(r'pubmed/([0-9]+)', ref_text)
                row['pubmed_ids'] = '|'.join(pmids)
                protein_rows.append(row)

    protein_df = pandas.DataFrame.from_dict(protein_rows)

    return drugbank_df, drugbank_slim_df,protein_rows